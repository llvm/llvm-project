#include <gpuintrin.h>
#include <stdint.h>

__gpu_kernel void foo(uint32_t *out) {
  int x = __gpu_block_id(0) * __gpu_num_threads(0) + __gpu_thread_id(0);
  int xw = __gpu_num_blocks(0) * __gpu_num_threads(0);
  int y = __gpu_block_id(1) * __gpu_num_threads(1) + __gpu_thread_id(1);
  int yw = __gpu_num_blocks(1) * __gpu_num_threads(1);
  int z = __gpu_block_id(2) * __gpu_num_threads(2) + __gpu_thread_id(2);
  int offset = (z * yw * xw) + (y * xw) + x;
  out[offset] = offset;
}
