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

class RangeFromBatchMinMax:
    def __call__(self, batch, tensor):
        minv = min(batch.flatten())
        maxv = max(batch.flatten())
        return minv, maxv

class RangeFromBatchMinMax98:
    def __call__(self, batch, tensor):
        batch_s = sorted(batch.flatten())
        assert(batch.size > 100)
        minv = batch_s[round(len(batch_s)*0.01)]
        maxv = batch_s[round(len(batch_s)*0.99)]
        return minv, maxv

class RangeFromBatchMinMax90:
    def __call__(self, batch, tensor):
        batch_s = sorted(batch.flatten())
        assert(batch.size > 100)
        minv = batch_s[round(len(batch_s)*0.05)]
        maxv = batch_s[round(len(batch_s)*0.95)]
        return minv, maxv

class RangeFromBatchMinMax80:
    def __call__(self, batch, tensor):
        batch_s = sorted(batch.flatten())
        assert(batch.size > 100)
        minv = batch_s[round(len(batch_s)*0.1)]
        maxv = batch_s[round(len(batch_s)*0.9)]
        return minv, maxv
