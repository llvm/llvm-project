// agx-cost: --uniform 0=0x1000000 --uniform 1=0 --map 0x1000000=@data/rowptr64.bin --uniform 2=0x2000000 --uniform 3=0 --alloc 0x2000000=1048576 --uniform 4=0x3000000 --uniform 5=0 --alloc 0x3000000=1048576 --uniform 6=0x4000000 --uniform 7=0 --alloc 0x4000000=16384 --uniform 8=0x5000000 --uniform 9=0 --alloc 0x5000000=16384 --uniform 10=4096 --wg 256
// Sparse: CSR y = A x, a SIMD-group per row (64 entries per row, as cuSPARSE's vector kernel).
#include "bench.h"
extern "C" __global__ void spmv_csr_vector(const int *rowptr, const int *col, const float *val, const float *x, float *y, int nrows) {
  int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= nrows)
    return;
  float s = 0.0f;
  for (int k = rowptr[r] + lane; k < rowptr[r + 1]; k += 32)
    s += val[k] * x[col[k]];
  s = warp_sum(s);
  if (lane == 0)
    y[r] = s;
}
