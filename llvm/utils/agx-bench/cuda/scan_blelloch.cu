// agx-cost: --wg 256
// Scan: work-efficient (Blelloch) exclusive prefix sum of 512 ints in threadgroup memory.
#include "bench.h"
extern "C" __global__ void scan_blelloch(const int *in, int *out) {
  __shared__ int t[512];
  int i = threadIdx.x, base = blockIdx.x * 512;
  t[2 * i] = in[base + 2 * i];
  t[2 * i + 1] = in[base + 2 * i + 1];
  int off = 1;
  for (int d = 256; d > 0; d >>= 1) {
    sync();
    if (i < d) {
      int a = off * (2 * i + 1) - 1, b = off * (2 * i + 2) - 1;
      t[b] += t[a];
    }
    off *= 2;
  }
  if (i == 0)
    t[511] = 0;
  for (int d = 1; d < 512; d *= 2) {
    off >>= 1;
    sync();
    if (i < d) {
      int a = off * (2 * i + 1) - 1, b = off * (2 * i + 2) - 1;
      int v = t[a];
      t[a] = t[b];
      t[b] += v;
    }
  }
  sync();
  out[base + 2 * i] = t[2 * i];
  out[base + 2 * i + 1] = t[2 * i + 1];
}
