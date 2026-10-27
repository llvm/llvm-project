// agx-cost: --uniform 4=32000 --wg 256
// LLM sampling: greedy argmax over a 32000-entry vocabulary (value / index pairs through shuffles and threadgroup memory).
#include "bench.h"
extern "C" __global__ void argmax(const float *logits, int *out, int n) {
  __shared__ float sv[32];
  __shared__ int si[32];
  float best = -3.0e38f;
  int bi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = logits[(long)blockIdx.x * n + i];
    if (v > best) { best = v; bi = i; }
  }
  for (int d = 16; d > 0; d >>= 1) {
    float v = shfl_xor(best, d);
    int j = shfl_xor_i(bi, d);
    if (v > best || (v == best && j < bi)) { best = v; bi = j; }
  }
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) { sv[w] = best; si[w] = bi; }
  sync();
  if (w == 0) {
    best = l < (blockDim.x >> 5) ? sv[l] : -3.0e38f;
    bi = l < (blockDim.x >> 5) ? si[l] : 0;
    for (int d = 16; d > 0; d >>= 1) {
      float v = shfl_xor(best, d);
      int j = shfl_xor_i(bi, d);
      if (v > best || (v == best && j < bi)) { best = v; bi = j; }
    }
    if (l == 0)
      out[blockIdx.x] = bi;
  }
}
