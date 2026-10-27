// agx-cost: --uniform 6=4096 --uniform 7=1 --grid 256 --wg 256
// Scientific f64: double precision dot product (grid-stride, software double on AGX), SIMD-group sums of doubles.
#include "bench.h"
extern "C" __global__ void ddot(const double *x, const double *y, double *out, int n) {
  __shared__ double t[8];
  double s = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    s += x[i] * y[i];
  for (int d = 16; d > 0; d >>= 1)
    s += shfl_xor_d(s, d);
  if ((threadIdx.x & 31) == 0)
    t[threadIdx.x >> 5] = s;
  sync();
  if (threadIdx.x == 0) {
    double r = 0.0;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w)
      r += t[w];
    out[blockIdx.x] = r;
  }
}
