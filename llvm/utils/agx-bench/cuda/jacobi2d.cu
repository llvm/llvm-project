// agx-cost: --uniform 4=1024 --uniform 5=1024 --wg 16,16 --wg-id 3,3
// Stencil: 5-point Jacobi step on a 1024 x 1024 grid with an 18 x 18 threadgroup tile (halo loads by edge threads).
#include "bench.h"
extern "C" __global__ void jacobi2d(float *out, const float *in, int nx, int ny) {
  __shared__ float t[18][18];
  int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
  bool inside = x < nx && y < ny;
  if (inside) {
    t[ty][tx] = in[y * nx + x];
    if (threadIdx.x == 0 && x > 0) t[ty][0] = in[y * nx + x - 1];
    if (threadIdx.x == 15 && x + 1 < nx) t[ty][17] = in[y * nx + x + 1];
    if (threadIdx.y == 0 && y > 0) t[0][tx] = in[(y - 1) * nx + x];
    if (threadIdx.y == 15 && y + 1 < ny) t[17][tx] = in[(y + 1) * nx + x];
  }
  sync();
  if (inside && x > 0 && y > 0 && x + 1 < nx && y + 1 < ny)
    out[y * nx + x] = 0.25f * (t[ty][tx - 1] + t[ty][tx + 1] + t[ty - 1][tx] + t[ty + 1][tx]);
}
