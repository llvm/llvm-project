// agx-cost: --uniform 8=512 --uniform 9=0x3DB504F3 --wg 128
// LLM inference: one decode step of attention for a 128-dim head over 512 cached f16 keys / values
// (dot products as SIMD-group sums, online softmax per SIMD-group, merged through threadgroup memory).
#include "bench.h"
extern "C" __global__ void attn_decode(const half *q, const half *K, const half *V, float *out, int seq, float scale) {
  __shared__ float wm[4], ws[4], wacc[4][128];
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  const half *qh = q + blockIdx.x * 128;
  float qv[4];
  for (int k = 0; k < 4; ++k)
    qv[k] = h2f(qh[lane * 4 + k]);
  float m = -3.0e38f, s = 0.0f, acc[4] = {0, 0, 0, 0};
  for (int t = w; t < seq; t += nw) {
    const half *kr = K + ((long)blockIdx.x * seq + t) * 128 + lane * 4;
    float d = 0.0f;
    for (int k = 0; k < 4; ++k)
      d += qv[k] * h2f(kr[k]);
    d = warp_sum(d) * scale;
    float m2 = __builtin_fmaxf(m, d), c = fexp(m - m2), p = fexp(d - m2);
    s = s * c + p;
    const half *vr = V + ((long)blockIdx.x * seq + t) * 128 + lane * 4;
    for (int k = 0; k < 4; ++k)
      acc[k] = acc[k] * c + p * h2f(vr[k]);
    m = m2;
  }
  if (lane == 0) { wm[w] = m; ws[w] = s; }
  for (int k = 0; k < 4; ++k)
    wacc[w][lane * 4 + k] = acc[k];
  sync();
  if (w == 0) {
    float M = -3.0e38f;
    for (int i = 0; i < nw; ++i)
      M = __builtin_fmaxf(M, wm[i]);
    float S = 0.0f, o[4] = {0, 0, 0, 0};
    for (int i = 0; i < nw; ++i) {
      float c = fexp(wm[i] - M);
      S += ws[i] * c;
      for (int k = 0; k < 4; ++k)
        o[k] += wacc[i][lane * 4 + k] * c;
    }
    for (int k = 0; k < 4; ++k)
      out[blockIdx.x * 128 + lane * 4 + k] = o[k] / S;
  }
}
