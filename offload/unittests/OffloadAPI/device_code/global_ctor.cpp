#include <gpuintrin.h>
#include <stdint.h>

extern "C" {

uint32_t global[64];

[[gnu::constructor(202)]] void ctorc() {
  for (unsigned I = 0; I < 64; I++)
    global[I] += 20;
}

[[gnu::constructor(200)]] void ctora() {
  for (unsigned I = 0; I < 64; I++)
    global[I] = 40;
}

[[gnu::constructor(201)]] void ctorb() {
  for (unsigned I = 0; I < 64; I++)
    global[I] *= 2;
}

__gpu_kernel void global_ctor(uint32_t *out) {
  global[__gpu_thread_id(0)] += __gpu_thread_id(0);
  out[__gpu_thread_id(0) + (__gpu_num_threads(0) * __gpu_block_id(0))] =
      global[__gpu_thread_id(0)];
}
} // extern "C"
