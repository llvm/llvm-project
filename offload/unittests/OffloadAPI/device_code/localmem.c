#include <gpuintrin.h>
#include <stdint.h>

extern __gpu_local uint32_t shared_mem[];

__gpu_kernel void localmem(uint32_t *out) {
  shared_mem[__gpu_thread_id(0)] = __gpu_thread_id(0);
  shared_mem[__gpu_thread_id(0)] *= 2;
  out[__gpu_thread_id(0) + (__gpu_num_threads(0) * __gpu_block_id(0))] =
      shared_mem[__gpu_thread_id(0)];
}
