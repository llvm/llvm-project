// agx-cost: --uniform 4=1920 --uniform 5=1080 --wg 16,16 --wg-id 10,10
// Image processing: 3 x 3 Gaussian blur of an RGBA8 image (byte loads, conversions, clamped edges).
#include "bench.h"
extern "C" __global__ void blur_rgba8(uc4 *out, const uc4 *in, int w, int h) {
  int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  if (x >= w || y >= h)
    return;
  const float k[3] = {0.25f, 0.5f, 0.25f};
  float r = 0, g = 0, b = 0, a = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      int xx = min(max(x + dx, 0), w - 1), yy = min(max(y + dy, 0), h - 1);
      uc4 p = in[yy * w + xx];
      float c = k[dx + 1] * k[dy + 1];
      r += c * p.x; g += c * p.y; b += c * p.z; a += c * p.w;
    }
  out[y * w + x] = uc4{(uchar)(r + 0.5f), (uchar)(g + 0.5f), (uchar)(b + 0.5f), (uchar)(a + 0.5f)};
}
