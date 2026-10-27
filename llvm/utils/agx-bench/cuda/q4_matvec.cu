// agx-cost: --uniform 6=4096 --wg 256
// LLM inference: y = W x with W in llama.cpp q4_0 blocks (f16 scale + 16 bytes of nibbles), a SIMD-group per row.
#include "bench.h"
struct q4_0 { half d; uchar qs[16]; };
extern "C" __global__ void q4_matvec(const q4_0 *W, const float *x, float *y, int ncols) {
  int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  int nb = ncols / 32;
  const q4_0 *r = W + (long)row * nb;
  float s = 0.0f;
  for (int b = lane; b < nb; b += 32) {
    const q4_0 *blk = r + b;
    const float *xb = x + b * 32;
    float t = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      uchar q = blk->qs[j];
      t += ((q & 15) - 8) * xb[j] + ((q >> 4) - 8) * xb[j + 16];
    }
    s += h2f(blk->d) * t;
  }
  s = warp_sum(s);
  if (lane == 0)
    y[row] = s;
}
