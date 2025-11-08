#include <gpuintrin.h>
#include <stdint.h>

extern "C" {

uint32_t global[64];

[[gnu::destructor]] void dtor() {
  for (unsigned I = 0; I < 64; I++)
    global[I] = 1;
}

__gpu_kernel void global_dtor() {
  // no-op
}
} // extern "C"
