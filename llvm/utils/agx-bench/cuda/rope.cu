// agx-cost: --uniform 2=128 --uniform 3=32 --uniform 4=77 --uniform 5=0x41549A78 --wg 256
// LLM inference: rotary position embedding of 32 heads x 128 dims in place (powers, sin / cos per pair).
#include "bench.h"
extern "C" __global__ void rope(float *q, int head_dim, int n_heads, int pos, float log2_base) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  int half_dim = head_dim / 2;
  if (p >= n_heads * half_dim)
    return;
  int h = p / half_dim, i = p % half_dim;
  float inv = __nvvm_ex2_approx_f(-log2_base * (2.0f * i / head_dim));
  float a = pos * inv, c = fcos(a), s = fsin(a);
  float *v = q + h * head_dim + 2 * i;
  float x0 = v[0], x1 = v[1];
  v[0] = x0 * c - x1 * s;
  v[1] = x0 * s + x1 * c;
}
