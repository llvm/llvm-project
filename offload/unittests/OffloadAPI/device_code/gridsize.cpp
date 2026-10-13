#include <gpuintrin.h>
#include <stdint.h>

extern "C" __gpu_kernel void gridsize(uint32_t *nblocks, uint32_t *nthreads) {
  if (__gpu_block_id(0) == 0 && __gpu_block_id(1) == 0 &&
      __gpu_block_id(2) == 0 && __gpu_thread_id(0) == 0 &&
      __gpu_thread_id(1) == 0 && __gpu_thread_id(2) == 0) {
    for (int i = 0; i < 3; ++i) {
      nblocks[i] = __gpu_num_blocks(i);
      nthreads[i] = __gpu_num_threads(i);
    }
  }
}
