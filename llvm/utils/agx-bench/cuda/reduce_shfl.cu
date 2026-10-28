// agx-cost: --uniform 4=65536 --uniform 5=16 --grid 4096 --wg 256
// Reduction: grid-stride sum with SIMD-group shuffles and one device atomic per SIMD-group (modern CUDA style).
#include "bench.h"
extern "C" __global__ void reduce_shfl(const float *x, float *out, int n) {
  float s = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    s += x[i];
  s = warp_sum(s);
  if ((threadIdx.x & 31) == 0)
    atomic_add(out, s);
}
