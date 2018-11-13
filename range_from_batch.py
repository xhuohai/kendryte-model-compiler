
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
