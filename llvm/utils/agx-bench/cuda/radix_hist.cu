// agx-cost: --uniform 4=16384 --uniform 5=8 --uniform 6=4 --grid 1024 --wg 256
// Sort: one radix-sort pass's digit histogram (4-bit digits, threadgroup atomics, then device atomics).
#include "bench.h"
extern "C" __global__ void radix_hist(const uint *keys, int *hist, int n, int shift) {
  __shared__ int h[16];
  if (threadIdx.x < 16)
    h[threadIdx.x] = 0;
  sync();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    atomic_add_i(&h[(keys[i] >> shift) & 15], 1);
  sync();
  if (threadIdx.x < 16)
    atomic_add_i(&hist[threadIdx.x * gridDim.x + blockIdx.x], h[threadIdx.x]);
}
