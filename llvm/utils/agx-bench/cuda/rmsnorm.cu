// agx-cost: --uniform 6=4096 --uniform 7=0x3727C5AC --wg 256
// LLM inference: RMSNorm of a 4096-float row per threadgroup (two passes, threadgroup reduction).
#include "bench.h"
extern "C" __global__ void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
  __shared__ float tmp[32];
  const float *row = x + (long)blockIdx.x * n;
  float s = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    s += row[i] * row[i];
  s = block_sum(s, tmp);
  float r = frsqrt(s / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    out[(long)blockIdx.x * n + i] = row[i] * r * w[i];
}
