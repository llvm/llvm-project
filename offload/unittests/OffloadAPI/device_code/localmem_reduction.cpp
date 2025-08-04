#include <gpuintrin.h>
#include <stdint.h>

extern __gpu_local uint32_t shared_mem[];

extern "C" __gpu_kernel void localmem_reduction(uint32_t *out) {
  shared_mem[__gpu_thread_id(0)] = 2;

  __gpu_sync_threads();

  if (__gpu_thread_id(0) == 0) {
    out[__gpu_block_id(0)] = 0;
    for (uint32_t i = 0; i < __gpu_num_threads(0); i++)
      out[__gpu_block_id(0)] += shared_mem[i];
  }
}
