// agx-cost: --uniform 6=4096 --wg 256
// LLM inference: token embedding gather (row index from memory, float4 copy of a 4096-float row).
#include "bench.h"
extern "C" __global__ void embedding(f4 *out, const int *ids, const f4 *table, int dim) {
  int id = ids[blockIdx.x];
  const f4 *src = table + (long)id * (dim / 4);
  f4 *dst = out + (long)blockIdx.x * (dim / 4);
  for (int i = threadIdx.x; i < dim / 4; i += blockDim.x)
    dst[i] = src[i];
}
