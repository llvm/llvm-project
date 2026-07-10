#include <gpuintrin.h>

extern "C" __gpu_kernel void multiargs(char A, int *B, short C) {
  B[__gpu_thread_id(0)] = A + C + __gpu_thread_id(0);
}
