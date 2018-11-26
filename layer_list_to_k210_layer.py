'''
 * Copyright 2018 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 '''

import math

import tensor_list_to_layer_list
import numpy as np


def hotfix_magic_1(eight_bit_mode):
    if eight_bit_mode:
        return 100000000.0 / 3
    else:
        return 100000000.0 / 3


def log_next_pow_of_2(value):
    ret = 0
    while value > 1 or value <= -1:
        value = value / 2
        ret = ret + 1

    return ret, value


def pow_next_log_of_2(value, bound_shift, shift_max_shift=4):
    ret = 0
    shift_max = 1 << shift_max_shift
    while value >= -(1 << (bound_shift - 2)) and value < (1 << (bound_shift - 2)) \
            and value != 0 and ret < (shift_max - 1):
        value = value * 2
        ret = ret + 1

    return ret, value


def signed_to_hex(value, width):
    return hex(int((1 << width) + value) % (1 << width))


class K210Conv:
    def __init__(self, weights, input_tensor_name, depth_wise_layer, eight_bit_mode, xy_shape, xw_minmax):
        self.weights = weights
        self.weights_shape = self.weights.shape
        self.input_shape, self.output_shape = xy_shape
        xmin, xmax, wmin, wmax = xw_minmax
        self.stride = 1 # layer.config['stride']
        self.depth_wise_layer = depth_wise_layer #isinstance(layer, tensor_list_to_layer_list.LayerDepthwiseConvolutional)
        self.eight_bit_mode = eight_bit_mode

        self.x_range = xmax - xmin
        self.x_bias = xmin
        assert (self.x_range > 0)

        self.w_range = wmax - wmin
        self.w_bias = wmin
        assert (self.w_range > 0)

        if self.input_shape[1:2] != self.output_shape[1:2]:
            # raise ValueError('conv2d {} should use padding=SAME'.format(input_tensor_name))
            print('[error]', 'conv2d {} should use padding=SAME'.format(input_tensor_name))
            self.input_shape = list(self.input_shape)
            self.input_shape[1] = self.output_shape[1]
            self.input_shape[2] = self.output_shape[2]

        if self.input_shape[1] < 4:
            tensor_height = self.input_shape[1]
            raise ValueError('feature map required height>4 which {} height is {}'.format(input_tensor_name, tensor_height))

    @staticmethod
    def q(value, scale, bias):
        return (value - bias) / scale

    def para_mult_loads(self, weights_shape, output_shape, kernel_size):
        weight_buffer_size = 2 * 9 * 4096
        weights_ich = int(weights_shape[2])
        weights_och = int(weights_shape[3])
        weight_data_size = 1 if self.eight_bit_mode else 2

        if self.depth_wise_layer:
            o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * weight_data_size
        else:
            o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * int(weights_shape[2]) * weight_data_size

        if int(weights_shape[0]) == 1:
            o_ch_weights_size_pad = math.ceil(o_ch_weights_size / 8) * 9
        else:
            o_ch_weights_size_pad = o_ch_weights_size
            assert (int(weights_shape[0]) == 3)

        if kernel_size == 3:
            load_time = math.ceil(weights_och / math.floor(4096 * 2 / weight_data_size / weights_ich))
        elif kernel_size == 1:
            load_time = math.ceil(weights_och / math.floor(4096 * 8 * 2 / weight_data_size / weights_ich))
        else:
            load_time = None
            assert (None)

        o_ch_num = int(output_shape[3])
        o_ch_num_coef = math.floor(weight_buffer_size / o_ch_weights_size_pad)

        if self.eight_bit_mode:
            half_weight_buffer_size = weight_buffer_size / 2
            while True:
                last_ch_idx = (o_ch_num - 1) % o_ch_num_coef
                last_addr_end = (last_ch_idx + 1) * o_ch_weights_size_pad
                if last_addr_end < half_weight_buffer_size:
                    break

                o_ch_num_coef = o_ch_num_coef - 1
                load_time = math.ceil(o_ch_num / o_ch_num_coef)
                if o_ch_num_coef <= 0:
                    assert ('cannot fix last_addr_end to first half part')

        assert (load_time <= 64)

        o_ch_num_coef = min(o_ch_num_coef, o_ch_num)
        para_size = o_ch_num_coef * o_ch_weights_size
        return load_time, para_size, o_ch_num_coef

    def to_k210(self):
        input_shape = self.input_shape
        output_shape = self.output_shape
        weights_shape = self.weights_shape
        weights = self.weights
        stride = self.stride

        weight_data_size = 1 if self.eight_bit_mode else 2
        kernel_size = int(weights_shape[0])

        # img i
        i_row_wid = int(input_shape[2])
        i_col_high = int(input_shape[1])
        coef_group = 1 if i_row_wid > 32 else (2 if i_row_wid > 16 else 4)
        row_switch_addr = math.ceil(i_row_wid / 64)
        channel_switch_addr = i_col_high * row_switch_addr
        # conv
        depth_wise_layer = 1 if self.depth_wise_layer else 0
        kernel_type = {1: 0, 3: 1}[kernel_size]
        pad_type = 0
        load_coor = 1

        first_stride = 0 if stride == 1 else 1
        assert (256 > (i_col_high if first_stride == 0 else i_col_high / 2))

        load_time, para_size, o_ch_num_coef = self.para_mult_loads(weights_shape, output_shape, kernel_size)

        x_qmax = 255
        w_qmax = (1 << (8 * weight_data_size)) - 1
        bias_x, scale_x = self.x_bias, self.x_range / x_qmax
        bias_w, scale_w = self.w_bias, self.w_range / w_qmax

        bx_div_sx = bias_x / scale_x
        bw_div_sw = bias_w / scale_w

        shr_x, arg_x = pow_next_log_of_2(bw_div_sw, 24)
        shr_w, arg_w = pow_next_log_of_2(bx_div_sx, 24)
        arg_add = kernel_size * kernel_size * bw_div_sw * bx_div_sx
        pad_value = -bx_div_sx
        swsx = scale_w * scale_x

        weight_q = ((weights - bias_w) / scale_w).transpose([3, 2, 0, 1])
        para_start_addr = [int(round(item)) for item in np.reshape(weight_q, (np.product(weight_q.shape),))]

        return {
            'swsx': swsx,
            'coef_group': coef_group,
            'channel_switch_addr': channel_switch_addr,
            'depth_wise_layer': depth_wise_layer,
            'o_ch_num_coef': o_ch_num_coef,
            'i_row_wid': i_row_wid,
            'i_col_high': i_col_high,
            'kernel_type': kernel_type,
            'pad_type': pad_type,
            'first_stride': first_stride,
            'pad_value': pad_value,
            'load_coor': load_coor,
            'load_time': load_time,
            'para_size': para_size,
            'para_start_addr': para_start_addr,
            'row_switch_addr': row_switch_addr,
            'shr_w': shr_w,
            'shr_x': shr_x,
            'arg_w': arg_w,
            'arg_x': arg_x,
            'arg_add': arg_add
        }


class K210BN:
    def __init__(self, mean, var, gamma, beta, epsilon, eight_bit_mode):
        self.mean = mean
        self.var = var
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.eight_bit_mode = eight_bit_mode

    @staticmethod
    def get_bn(scale, bias):
        norm_shift, norm_mul = pow_next_log_of_2(scale, 24)
        return {'norm_mul': signed_to_hex(norm_mul, 24), 'norm_add': signed_to_hex(bias, 32), 'norm_shift': norm_shift}

    def to_k210(self, swsx=1):
        __hotfix_magic = hotfix_magic_1(self.eight_bit_mode)
        sqrt_var = np.sqrt(self.var + self.epsilon)
        scale = swsx * self.gamma / sqrt_var * __hotfix_magic
        bias = (self.beta - self.gamma * self.mean / sqrt_var) * __hotfix_magic

        load_para = 1
        bwsx_base_addr = [
            self.get_bn(s, b)
            for s, b in zip(scale.tolist(), bias.tolist())
        ]

        return locals()


class K210Act:
    def __init__(self, act_tensor, min_y, max_y, name, eight_bit_mode):
        self.name = name
        self.tensor = act_tensor
        self.eight_bit_mode = eight_bit_mode
        self.min_y = min_y
        self.max_y = max_y

    @staticmethod
    def leaky_relu(x):
        return x if x >= 0 else 0.1 * x

    @staticmethod
    def leaky_relu_inverse(y):
        return y if y >= 0 else 10 * y

    @staticmethod
    def relu_inverse(y):
        return y

    @staticmethod
    def relu6_inverse(y):
        return y

    @staticmethod
    def leaky_table(min_y, max_y):
        range_y = max_y - min_y
        y_table = [min_y + i * range_y / 15 for i in range(15)]
        y_table.append(max_y)
        if 0 not in y_table:
            y_table.append(0)
        y_table = sorted(y_table)
        x_table = [K210Act.leaky_relu_inverse(it) for it in y_table]
        dydx = [(y_table[i + 1] - y_table[i]) / (x_table[i + 1] - x_table[i]) for i in range(len(y_table) - 1)]
        return zip(x_table, y_table, dydx)

    @staticmethod
    def relu_table(min_y, max_y):
        range_y = max_y - min_y
        y_table = [min_y + i * range_y / 15 for i in range(15)]
        y_table.append(max_y)
        if 0 not in y_table:
            y_table.append(0)
        y_table = sorted(y_table)
        x_table = [K210Act.relu_inverse(it) for it in y_table]
        dydx = [(y_table[i + 1] - y_table[i]) / (x_table[i + 1] - x_table[i]) for i in range(len(y_table) - 1)]
        return zip(x_table, y_table, dydx)

    @staticmethod
    def relu6_table(min_y, max_y):
        range_y = max_y - min_y
        y_table = [min_y + i * range_y / 15 for i in range(15)]
        y_table.append(max_y)
        if 0 not in y_table:
            y_table.append(0)
        y_table = sorted(y_table)
        x_table = [K210Act.relu6_inverse(it) for it in y_table]
        dydx = [(y_table[i + 1] - y_table[i]) / (x_table[i + 1] - x_table[i]) for i in range(len(y_table) - 1)]
        return zip(x_table, y_table, dydx)

    @staticmethod
    def linear_table(min_y, max_y):
        range_y = max_y - min_y
        y_table = [min_y + i * range_y / 14 for i in range(14)]
        if 0 not in y_table:
            y_table.append(0)
        y_table.append(max_y)
        y_table = sorted(y_table)
        return zip(y_table, y_table, [1] * (len(y_table) - 1))

    @staticmethod
    def find_shift(dydx):
        ret_shift = 0
        while abs(dydx) < (1 << 14) and dydx > 0:
            dydx = dydx * 2
            ret_shift = ret_shift + 1
        return ret_shift, dydx

    @staticmethod
    def table_to_act(act_table, min_y, max_y, eight_bit_mode):
        def act_table_aux(x, y, dydx):
            y_scale = (max_y - min_y) / 255
            y_bias = min_y
            x_fix = x * hotfix_magic_1(eight_bit_mode)
            y_fix = (y - y_bias) / y_scale
            dydx_fix = dydx / y_scale / hotfix_magic_1(eight_bit_mode)

            yf_q = round(y_fix)
            yf_err = y_fix - yf_q
            xfy = x_fix - yf_err / dydx_fix
            return xfy, yf_q, dydx_fix

        act_table = [(0x800000000, 0, 0)] + [act_table_aux(x, y, dydx) for x, y, dydx in act_table]

        def ret_aux(x, y, dydx):
            dxss, dys = K210Act.find_shift(dydx)
            assert (dys >= 0)
            return {'x': int(round(x)), 'y': int(round(y)), 'dxs': dxss, 'dy': int(round(dys))}

        return [ret_aux(x, y, dydx) for x, y, dydx in act_table]

    def to_k210(self):
        act_tab = None
        if self.name == 'leaky':
            act_tab = list(K210Act.leaky_table(self.min_y, self.max_y))
        elif self.name == 'Relu':
            act_tab = list(K210Act.relu_table(self.min_y, self.max_y))
        elif self.name == 'Relu6':
            act_tab = list(K210Act.relu6_table(self.min_y, self.max_y))
        elif self.name == 'linear':
            act_tab = list(K210Act.linear_table(self.min_y, self.max_y))
        else:
            print(self.name, ' active is not supported.')
            assert (None)
        return {'active_addr': K210Act.table_to_act(list(act_tab), self.min_y, self.max_y, self.eight_bit_mode)[:16]}


class K210Pool:
    def __init__(self, layer, size, stride, sess=None, dataset=None):
        self.size = size
        self.stride = stride
        self.tensor = layer.tensor_pool
        if self.size == 2 and self.tensor.op.inputs[0].shape[3] % 2 != 0:
            if self.tensor.op.get_attr('padding') == b'SAME':
                raise ValueError("at {} unsupport padding mode SAME of pooling with size == 2".format(self.tensor.name))

    def to_k210(self):
        if self.tensor.op.type == 'MaxPool':
            return {'pool_type': {
                (2, 2): 1,
                (4, 2): 3,
                (2, 1): 9
            }[(self.size, self.stride)]}
        elif self.tensor.op.type == 'AvgPool':
            return {'pool_type': {
                (2, 2): 2,
                (4, 2): 4,
                (2, 1): 8
            }[(self.size, self.stride)]}
        else:
            return None


class K210Layer:
    def __init__(self, eight_bit_mode):
        self.conv = None
        self.bn = None
        self.act = None
        self.pool = None
        self.eight_bit_mode = eight_bit_mode

    @staticmethod
    def batch(iter, n=1):
        l = len(iter)
        for ndx in range(0, l, n):
            yield iter[ndx:min(ndx + n, l)]

    def to_k210(self, idx):
        if self.pool is not None:
            output_shape = self.pool.tensor.shape
        else:
            output_shape = self.conv.output_shape

        weights_shape = self.conv.weights_shape
        input_shape = self.conv.input_shape
        i_row_wid = int(input_shape[1])
        img_data_size = 1

        coef_group = 1 if i_row_wid > 32 else (2 if i_row_wid > 16 else 4)

        # io
        i_ch_num = int(weights_shape[2])
        o_ch_num = int(output_shape[3])
        # img o
        o_row_wid = int(output_shape[2])
        o_col_high = int(output_shape[1])
        wb_group = 1 if o_row_wid > 32 else (2 if o_row_wid > 16 else 4)
        wb_row_switch_addr = math.ceil(o_row_wid / 64)
        wb_channel_switch_addr = o_col_high * wb_row_switch_addr
        channel_byte_num = o_row_wid * o_col_high

        int_en = 0
        image_src_addr = None
        image_dst_addr = None
        dma_total_byte = o_row_wid * o_col_high * o_ch_num
        dma_burst_size = 0xf
        send_data_out = 0
        return locals()


def make_k210_layer(sess, dataset, buffer, last_min, last_max, eight_bit_mode, range_from_batch):
    cur_k210 = K210Layer(eight_bit_mode)
    conv_layer = None

    if isinstance(buffer[-1], tensor_list_to_layer_list.LayerConvolutional) \
            or isinstance(buffer[-1], tensor_list_to_layer_list.LayerDepthwiseConvolutional):
        conv_layer = buffer.pop()

        conv_input_shape = sess.run(conv_layer.tensor_conv_x, dataset).shape
        conv_output_shape = sess.run(conv_layer.tensor_conv_y, dataset).shape
        wmin, wmax, _ = range_from_batch(sess, conv_layer.tensor_conv_w, dataset, is_weights=True)
        cur_k210.conv = K210Conv(
            conv_layer.weights, conv_layer.tensor_conv_x.name,
            isinstance(conv_layer, tensor_list_to_layer_list.LayerDepthwiseConvolutional),
            eight_bit_mode, [conv_input_shape, conv_output_shape],
            [last_min, last_max, wmin, wmax]
        )
        if int(conv_layer.config['batch_normalize']) == 1:
            cur_k210.bn = K210BN(
                conv_layer.batch_normalize_moving_mean,
                conv_layer.batch_normalize_moving_variance,
                conv_layer.batch_normalize_gamma,
                conv_layer.batch_normalize_beta,
                conv_layer.batch_normalize_epsilon,
                eight_bit_mode
            )
        else:
            bias_shape = conv_layer.bias.shape
            cur_k210.bn = K210BN(0, 1, np.ones(bias_shape), conv_layer.bias, 0, eight_bit_mode)

        tensor_act = conv_layer.tensor_activation
        act_min_y, act_max_y, _ = range_from_batch(sess, tensor_act, dataset)
        cur_k210.act = K210Act(tensor_act, act_min_y, act_max_y, conv_layer.config['activation'],
                               eight_bit_mode=eight_bit_mode)

    if len(buffer) > 0 and isinstance(buffer[-1], tensor_list_to_layer_list.LayerPool):
        pool_layer = buffer.pop()
        assert (isinstance(pool_layer, tensor_list_to_layer_list.LayerPool))
        # hotfix
        if pool_layer.config['stride'] == 1 and conv_layer.config['stride'] == 2:
            pool_layer.config['stride'] = 2

        cur_k210.pool = K210Pool(pool_layer, pool_layer.config['size'], pool_layer.config['stride'],
                                 sess, dataset)

    return cur_k210


def make_id_layer(base_tensor, min_v, max_v, eight_bit_mode, range_from_batch):
    o_ch = base_tensor.shape[1]
    input_tensor_shape = base_tensor.shape
    cur_k210 = K210Layer(eight_bit_mode)
    cur_k210.conv = K210Conv(
        weights=np.array([[[[1]]]]),
        input_tensor_name='<id_layer>',
        depth_wise_layer=True,
        eight_bit_mode=eight_bit_mode, xy_shape=[input_tensor_shape, input_tensor_shape],
        xw_minmax=[min_v, max_v, min_v, max_v]
    )
    cur_k210.bn = K210BN(0, 1, np.ones(o_ch), np.zeros(o_ch), eight_bit_mode)
    cur_k210.act = K210Act(base_tensor, min_v, max_v, 'linear', eight_bit_mode=eight_bit_mode)
    cur_k210.pool = None
    return cur_k210


def k210_layer_post_fix(klayer: K210Layer):
    return klayer


def gen_k210_layers(layers: [tensor_list_to_layer_list.LayerBase], sess, dataset, range_from_batch,
                    eight_bit_mode=False, input_min=0, input_max=1):
    buffer = list(layers)
    buffer.reverse()
    ret = []

    net = buffer.pop()
    assert (isinstance(net, tensor_list_to_layer_list.LayerNet))

    while len(buffer) != 0:
        if len(ret) > 0:
            last_act = ret[-1].act
            last_min = last_act.min_y
            last_max = last_act.max_y
        else:
            last_min = input_min
            last_max = input_max


        cur_k210 = make_k210_layer(
            sess=sess, dataset=dataset,
            buffer=buffer,
            last_min=last_min, last_max=last_max,
            eight_bit_mode=eight_bit_mode,
            range_from_batch=range_from_batch
        )

        cur_k210_fixed = k210_layer_post_fix(cur_k210)
        ret.append(cur_k210_fixed) if isinstance(cur_k210_fixed, K210Layer) else ret.extend(cur_k210_fixed)

    return ret
