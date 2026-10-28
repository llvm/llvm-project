// agx-cost: --uniform 4=16384 --uniform 5=4 --grid 1024 --wg 256
// Transformers: tanh-approximated GELU over f16 activations, half2 vectors in a grid-stride loop.
#include "bench.h"
extern "C" __global__ void gelu_f16(half2 *out, const half2 *x, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2; i += blockDim.x * gridDim.x) {
    half2 v = x[i], o;
    float a = h2f(v.x), b = h2f(v.y);
    o.x = f2h(0.5f * a * (1.0f + ftanh(0.7978845608f * (a + 0.044715f * a * a * a))));
    o.y = f2h(0.5f * b * (1.0f + ftanh(0.7978845608f * (b + 0.044715f * b * b * b))));
    out[i] = o;
  }
}
