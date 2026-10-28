// agx-cost: --uniform 6=64 --uniform 7=256 --wg 256
// Frameworks: index_add / scatter-add of 64 rows of 256 floats into rows chosen by an index tensor (float atomics).
#include "bench.h"
extern "C" __global__ void index_add(float *out, const int *idx, const float *src, int n, int dim) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= dim)
    return;
  for (int i = 0; i < n; ++i)
    atomic_add(&out[(long)idx[i] * dim + j], src[(long)i * dim + j]);
}
