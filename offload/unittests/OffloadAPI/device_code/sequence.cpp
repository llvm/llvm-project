#include <gpuintrin.h>
#include <stdint.h>

extern "C" __gpu_kernel void sequence(uint32_t idx, uint32_t *inout) {
  if (idx == 0)
    inout[idx] = 0;
  else if (idx == 1)
    inout[idx] = 1;
  else
    inout[idx] = inout[idx - 1] + inout[idx - 2];
}
