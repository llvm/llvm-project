// agx-cost: --wg 512
// Sort: bitonic sort of 1024 keys in threadgroup memory (compare-exchange, divergent swaps, many barriers).
#include "bench.h"
extern "C" __global__ void bitonic_smem(uint *keys) {
  __shared__ uint t[1024];
  int i = threadIdx.x;
  uint *k = keys + blockIdx.x * 1024;
  t[i] = k[i];
  t[i + 512] = k[i + 512];
  for (int size = 2; size <= 1024; size <<= 1)
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      sync();
      int p = 2 * i - (i & (stride - 1));
      bool up = (p & size) == 0;
      uint a = t[p], b = t[p + stride];
      if ((a > b) == up) {
        t[p] = b;
        t[p + stride] = a;
      }
    }
  sync();
  k[i] = t[i];
  k[i + 512] = t[i + 512];
}
