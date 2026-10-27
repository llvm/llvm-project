// agx-cost: --uniform 8=4096 --uniform 9=0x3727C5AC --wg 256
// Transformers: LayerNorm of a 4096-element f16 row (half2 loads, f32 statistics, affine f16 output).
#include "bench.h"
extern "C" __global__ void layernorm_f16(half2 *out, const half2 *x, const half2 *g, const half2 *b, int n, float eps) {
  __shared__ float tmp[32];
  int n2 = n / 2;
  const half2 *row = x + (long)blockIdx.x * n2;
  float s = 0.0f, ss = 0.0f;
  for (int i = threadIdx.x; i < n2; i += blockDim.x) {
    half2 v = row[i];
    float a = h2f(v.x), c = h2f(v.y);
    s += a + c;
    ss += a * a + c * c;
  }
  s = block_sum(s, tmp);
  sync();
  ss = block_sum(ss, tmp);
  float mean = s / n, r = frsqrt(ss / n - mean * mean + eps);
  for (int i = threadIdx.x; i < n2; i += blockDim.x) {
    half2 v = row[i], gg = g[i], bb = b[i], o;
    o.x = f2h((h2f(v.x) - mean) * r * h2f(gg.x) + h2f(bb.x));
    o.y = f2h((h2f(v.y) - mean) * r * h2f(gg.y) + h2f(bb.y));
    out[(long)blockIdx.x * n2 + i] = o;
  }
}
