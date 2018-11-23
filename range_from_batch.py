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
import numpy as np

class RangeFromBatchMinMax:
    def __call__(self, sess, tensor, dataset, is_weights=False):
        batch = sess.run(tensor, dataset)
        minv = min(batch.flatten())
        maxv = max(batch.flatten())
        return minv, maxv, batch

class RangeFromBatchMinMax98:
    def __call__(self, sess, tensor, dataset, is_weights=False):
        batch = sess.run(tensor, dataset)
        batch_s = sorted(batch.flatten())
        assert(batch.size > 100)
        minv = batch_s[round(len(batch_s)*0.01)]
        maxv = batch_s[round(len(batch_s)*0.99)]
        return minv, maxv, batch

class RangeFromBatchMinMax90:
    def __call__(self, sess, tensor, dataset, is_weights=False):
        batch = sess.run(tensor, dataset)
        batch_s = sorted(batch.flatten())
        assert(batch.size > 100)
        minv = batch_s[round(len(batch_s)*0.05)]
        maxv = batch_s[round(len(batch_s)*0.95)]
        return minv, maxv, batch

class RangeFromBatchMinMax80:
    def __call__(self, sess, tensor, dataset, is_weights=False):
        batch = sess.run(tensor, dataset)
        batch_s = sorted(batch.flatten())
        assert(batch.size > 100)
        minv = batch_s[round(len(batch_s)*0.1)]
        maxv = batch_s[round(len(batch_s)*0.9)]
        return minv, maxv, batch

class RangeFromBatchMeanMinsMaxs:
    def __call__(self, sess, tensor, dataset, is_weights=False):
        if is_weights:
            return RangeFromBatchMinMax()(sess, tensor,dataset,is_weights)
        else:
            batch = sess.run(tensor, dataset)
            n_batch = np.reshape(batch, [batch.shape[0], np.prod(batch.shape[1:])])
            minv = n_batch.min(axis=1).mean()
            maxv = n_batch.max(axis=1).mean()
            return minv, maxv, batch