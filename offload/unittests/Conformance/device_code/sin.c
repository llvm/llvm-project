#include <gpuintrin.h>
#include <math.h>

__gpu_kernel void kernel(double *out) { *out = sin(*out); }
