// agx-cost: --uniform 2=1234 --uniform 3=1048576 --wg 256
// Random numbers: Philox4x32-10 (as curand) to uniform floats, 4 per thread (32-bit multiply-high heavy).
#include "bench.h"
extern "C" __global__ void philox(f4 *out, uint seed, int n) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 4 >= (uint)n)
    return;
  uint c0 = i, c1 = 0, c2 = 0, c3 = 0, k0 = seed, k1 = 0xcafef00d;
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    uint h0 = __nvvm_mulhi_ui(0xD2511F53u, c0), l0 = 0xD2511F53u * c0;
    uint h1 = __nvvm_mulhi_ui(0xCD9E8D57u, c2), l1 = 0xCD9E8D57u * c2;
    c0 = h1 ^ c1 ^ k0; c1 = l1; c2 = h0 ^ c3 ^ k1; c3 = l0;
    k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
  }
  const float s = 2.3283064e-10f;
  out[i] = f4{c0 * s, c1 * s, c2 * s, c3 * s};
}
