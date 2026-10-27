// agx-cost: --uniform 2=1024 --uniform 3=1024 --uniform 4=256 --wg 16,16 --wg-id 26,34
// Divergent loops: Mandelbrot escape counts (a threadgroup in the seahorse valley: iteration counts differ per lane).
#include "bench.h"
extern "C" __global__ void mandelbrot(uchar *out, int w, int h, int maxit) {
  int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  if (x >= w || y >= h)
    return;
  // float(x) as 2^23 + x - 2^23: agx_cost does not emulate convert, which would leave every lane at -2 - 1.5i.
  float fx = __builtin_bit_cast(float, 0x4B000000 | x) - 8388608.0f;
  float fy = __builtin_bit_cast(float, 0x4B000000 | y) - 8388608.0f;
  float cr = -2.0f + fx * (3.0f / 1024), ci = -1.5f + fy * (3.0f / 1024), zr = 0.0f, zi = 0.0f;
  int it = 0;
  while (it < maxit && zr * zr + zi * zi < 4.0f) {
    float t = zr * zr - zi * zi + cr;
    zi = 2.0f * zr * zi + ci;
    zr = t;
    ++it;
  }
  out[y * w + x] = (uchar)it;
}
