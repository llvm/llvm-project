#include <gpuintrin.h>
#include <stdint.h>

extern "C" __gpu_kernel void single_counter(int32_t init_loop, int32_t addend,
                                            uint32_t *init_val, uint32_t *out) {

  if (__gpu_thread_id(0) != 0) {
    return;
  }

  volatile int32_t counter = init_loop;

  // We save the value at the beginning of the function in order to verify later
  // whether the previously submitted kernel has finished before the new
  // enqueue.
  uint32_t local_sum = *init_val;

  while (counter--) {
    local_sum += addend;
  }

  *out = local_sum;
}
