#include <gpuintrin.h>
#include <stdint.h>

#ifndef __SPIRV
extern __gpu_local uint32_t shared_mem[];
extern "C" __gpu_kernel void localmem(uint32_t *out) {
#else
extern "C" __gpu_kernel void localmem(uint32_t *out,
                                      __gpu_local uint32_t *shared_mem) {
#endif
  shared_mem[__gpu_thread_id(0)] = __gpu_thread_id(0);
  shared_mem[__gpu_thread_id(0)] *= 2;
  out[__gpu_thread_id(0) + (__gpu_num_threads(0) * __gpu_block_id(0))] =
      shared_mem[__gpu_thread_id(0)];
}
