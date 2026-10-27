// agx-cost: --uniform 6=65536 --uniform 7=16 --grid 4096 --wg 256
// Parallel primitives: stream compaction, an atomic counter per element kept (atomicAdd's result is the slot; the test keeps zeros, which is every element in agx_cost, where memory reads as zero).
#include "bench.h"
extern "C" __global__ void compact(float *out, int *count, const float *x, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    float v = x[i];
    if (v == 0.0f)
      out[atomic_add_i(count, 1)] = v;
  }
}
