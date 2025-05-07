#include <gpuintrin.h>

__gpu_kernel void foo(int *out) {
  out[__gpu_thread_id(0)] = __gpu_thread_id(0);
}
