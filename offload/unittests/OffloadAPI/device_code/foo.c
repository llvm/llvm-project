#include <gpuintrin.h>
#include <stdint.h>

__gpu_kernel void foo(uint32_t *out) {
  out[__gpu_thread_id(0)] = __gpu_thread_id(0);
}
