#include <gpuintrin.h>

extern "C" __gpu_kernel void byte(unsigned char c) { (void)c; }
