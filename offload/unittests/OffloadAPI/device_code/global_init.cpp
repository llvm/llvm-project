#include <gpuintrin.h>
#include <stdint.h>

extern "C" {

// Initialized, so it has contents in the program image (unlike `.bss`).
[[gnu::visibility("default")]]
uint32_t global_init[4] = {1, 2, 3, 4};

__gpu_kernel void read(uint32_t *out) {
  out[__gpu_thread_id(0)] = global_init[__gpu_thread_id(0) % 4];
}
} // extern "C"
