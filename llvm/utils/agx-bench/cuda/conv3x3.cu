// agx-cost: --uniform 6=64 --uniform 7=56 --uniform 8=56 --wg 16,16 --wg-id 1,1
// CNN: direct 3 x 3 convolution, NCHW, 64 input channels, an output pixel per thread (one output channel per grid z).
#include "bench.h"
extern "C" __global__ void conv3x3(float *out, const float *in, const float *wt, int C, int H, int W) {
  int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y, k = blockIdx.z;
  if (x >= W || y >= H)
    return;
  float s = 0.0f;
  for (int c = 0; c < C; ++c) {
    const float *ip = in + (long)c * H * W, *wp = wt + (k * C + c) * 9;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
      for (int dx = -1; dx <= 1; ++dx) {
        int yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
          s += ip[yy * W + xx] * wp[(dy + 1) * 3 + dx + 1];
      }
  }
  out[((long)k * H + y) * W + x] = s;
}
