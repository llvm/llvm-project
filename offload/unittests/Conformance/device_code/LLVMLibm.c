#include <gpuintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

typedef _Float16 float16;

__gpu_kernel void hypotf16Kernel(const float16 *X, float16 *Y, float16 *Out,
                                 size_t NumElements) {
  uint32_t Index =
      __gpu_num_threads_x() * __gpu_block_id_x() + __gpu_thread_id_x();

  if (Index < NumElements)
    Out[Index] = hypotf16(X[Index], Y[Index]);
}

__gpu_kernel void logfKernel(const float *X, float *Out, size_t NumElements) {
  uint32_t Index =
      __gpu_num_threads_x() * __gpu_block_id_x() + __gpu_thread_id_x();

  if (Index < NumElements)
    Out[Index] = logf(X[Index]);
}
