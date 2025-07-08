#include <gpuintrin.h>
#include <stdint.h>

[[clang::loader_uninitialized]]
uint32_t global[64];

__attribute__((constructor)) void ctor() {
  for (unsigned I = 0; I < 64; I++)
    global[I] = 100;
}

__gpu_kernel void global_ctor(uint32_t *out) {
  global[__gpu_thread_id(0)] += __gpu_thread_id(0);
  out[__gpu_thread_id(0) + (__gpu_num_threads(0) * __gpu_block_id(0))] =
      global[__gpu_thread_id(0)];
}
