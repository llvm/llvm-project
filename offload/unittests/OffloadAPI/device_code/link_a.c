#include <gpuintrin.h>
#include <stdint.h>

uint32_t global;

extern uint32_t funky();

__gpu_kernel void link_a(uint32_t *out) {
  out[0] = funky();
  out[1] = global;
}
