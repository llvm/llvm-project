// agx-cost: --uniform 6=8192 --uniform 7=4 --grid 1024 --wg 256
// LLM inference: SwiGLU (silu(gate) * up) on float4 vectors.
#include "bench.h"
extern "C" __global__ void swiglu(f4 *out, const f4 *gate, const f4 *up, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n / 4; i += blockDim.x * gridDim.x) {
    f4 g = gate[i], u = up[i], o;
    o.x = g.x * frcp(1.0f + fexp(-g.x)) * u.x;
    o.y = g.y * frcp(1.0f + fexp(-g.y)) * u.y;
    o.z = g.z * frcp(1.0f + fexp(-g.z)) * u.z;
    o.w = g.w * frcp(1.0f + fexp(-g.w)) * u.w;
    out[i] = o;
  }
}
