// agx-cost: --uniform 6=256 --uniform 7=256 --uniform 8=128 --wg 16,16
// Dense: hand-written SGEMM, 64 x 64 threadgroup tiles in threadgroup memory, a 4 x 4 register block per thread.
#include "bench.h"
extern "C" __global__ void sgemm_reg4x4(float *C, const float *A, const float *B, int M, int N, int K) {
  __shared__ float As[16][64], Bs[16][64];
  int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
  int row0 = blockIdx.y * 64, col0 = blockIdx.x * 64;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int l = tid; l < 16 * 64; l += 256) {
      int r = l / 16, c = l % 16;
      As[c][r] = A[(long)(row0 + r) * K + k0 + c];
      int r2 = l / 64, c2 = l % 64;
      Bs[r2][c2] = B[(long)(k0 + r2) * N + col0 + c2];
    }
    sync();
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      float a[4], b[4];
      for (int i = 0; i < 4; ++i) { a[i] = As[k][ty * 4 + i]; b[i] = Bs[k][tx * 4 + i]; }
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          acc[i][j] += a[i] * b[j];
    }
    sync();
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      C[(long)(row0 + ty * 4 + i) * N + col0 + tx * 4 + j] = acc[i][j];
}
