// agx-cost: --uniform 4=512 --uniform 5=0x3A83126F --wg 256
// Simulation: all-pairs n-body accelerations (float4 bodies tiled through threadgroup memory, rsqrt).
#include "bench.h"
extern "C" __global__ void nbody(const f4 *pos, f4 *acc, int n, float eps2) {
  __shared__ f4 tile[256];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  f4 p = pos[i];
  float ax = 0.0f, ay = 0.0f, az = 0.0f;
  for (int base = 0; base < n; base += 256) {
    tile[threadIdx.x] = pos[base + threadIdx.x];
    sync();
#pragma unroll 4
    for (int j = 0; j < 256; ++j) {
      f4 q = tile[j];
      float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
      float r = frsqrt(dx * dx + dy * dy + dz * dz + eps2);
      float f = q.w * r * r * r;
      ax += dx * f; ay += dy * f; az += dz * f;
    }
    sync();
  }
  acc[i] = f4{ax, ay, az, 0.0f};
}
