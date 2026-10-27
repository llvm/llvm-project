// agx-cost: --uniform 4=65536 --uniform 6=1 --uniform 8=2048 --uniform 10=64 --uniform 12=524288 --uniform 14=64 --uniform 15=32 --uniform 16=32 --wg 256 --wg-id 100,0,0
// Frameworks: PyTorch-style strided copy (NHWC -> NCHW): 64-bit offsets from a linear index by division / modulo by runtime sizes.
#include "bench.h"
extern "C" __global__ void permute4d(float *out, const float *in, long sn, long sc, long sh, long sw, long total, int C, int H, int W) {
  long base = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    long idx = base + u;
    if (idx >= total)
      return;
    long w = idx % W, t = idx / W;
    long h = t % H;
    t /= H;
    long c = t % C, n = t / C;
    out[idx] = in[n * sn + c * sc + h * sh + w * sw];
  }
}
