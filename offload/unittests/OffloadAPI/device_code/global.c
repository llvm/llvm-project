#include <gpuintrin.h>
#include <stdint.h>

[[gnu::visibility("default")]]
uint32_t global[64];

__gpu_kernel void write() {
  global[__gpu_thread_id(0)] = __gpu_thread_id(0);
  global[__gpu_thread_id(0)] *= 2;
}

__gpu_kernel void read(uint32_t *out) {
  out[__gpu_thread_id(0) + (__gpu_num_threads(0) * __gpu_block_id(0))] =
      global[__gpu_thread_id(0)];
}
