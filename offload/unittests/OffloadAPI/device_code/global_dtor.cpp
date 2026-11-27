#include <gpuintrin.h>
#include <stdint.h>

extern "C" {

uint32_t global[64];

// Host-visible location the destructor reports to, see set_dtor_target.
uint32_t *dtor_target = nullptr;

[[gnu::destructor]] void dtor() {
  for (unsigned I = 0; I < 64; I++)
    global[I] = 1;
  if (dtor_target)
    *dtor_target = 0xD7;
}

__gpu_kernel void set_dtor_target(uint32_t *Target) { dtor_target = Target; }

__gpu_kernel void global_dtor() {
  // no-op
}
} // extern "C"
