#include <gpuintrin.h>
#include <stdint.h>

// Words of dynamic local memory per thread, filled with values unique to the
// work-group so that work-groups sharing the memory are detected.
#define WORDS_PER_THREAD 16

extern __gpu_local uint32_t shared_mem[];

extern "C" __gpu_kernel void localmem_groups(uint32_t *out) {
  uint32_t Tid = __gpu_thread_id(0);
  uint32_t NumThreads = __gpu_num_threads(0);
  uint32_t Base = __gpu_block_id(0) * NumThreads * WORDS_PER_THREAD;

  for (uint32_t I = 0; I < WORDS_PER_THREAD; ++I)
    shared_mem[Tid * WORDS_PER_THREAD + I] = Base + Tid * WORDS_PER_THREAD + I;

  __gpu_sync_threads();

  // Copy out the words of the next thread of the work-group.
  uint32_t Other = (Tid + 1) % NumThreads;
  for (uint32_t I = 0; I < WORDS_PER_THREAD; ++I)
    out[Base + Tid * WORDS_PER_THREAD + I] =
        shared_mem[Other * WORDS_PER_THREAD + I];
}
