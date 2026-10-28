// agx-cost: --uniform 4=2048 --wg 256
// LLM inference: softmax of a 2048-float row, online max / sum in one pass, then the normalized write.
#include "bench.h"
extern "C" __global__ void softmax_online(float *out, const float *x, int n) {
  __shared__ float sm[32], ss[32];
  const float *row = x + (long)blockIdx.x * n;
  float m = -3.0e38f, s = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = row[i], m2 = __builtin_fmaxf(m, v);
    s = s * fexp(m - m2) + fexp(v - m2);
    m = m2;
  }
  for (int d = 16; d > 0; d >>= 1) {
    float m2 = shfl_xor(m, d), s2 = shfl_xor(s, d), mm = __builtin_fmaxf(m, m2);
    s = s * fexp(m - mm) + s2 * fexp(m2 - mm);
    m = mm;
  }
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) { sm[w] = m; ss[w] = s; }
  sync();
  m = l < (blockDim.x >> 5) ? sm[l] : -3.0e38f;
  s = l < (blockDim.x >> 5) ? ss[l] : 0.0f;
  for (int d = 16; d > 0; d >>= 1) {
    float m2 = shfl_xor(m, d), s2 = shfl_xor(s, d), mm = __builtin_fmaxf(m, m2);
    s = s * fexp(m - mm) + s2 * fexp(m2 - mm);
    m = mm;
  }
  float r = 1.0f / s;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    out[(long)blockIdx.x * n + i] = fexp(row[i] - m) * r;
}
