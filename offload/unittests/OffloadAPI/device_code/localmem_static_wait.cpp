#include <gpuintrin.h>
#include <stdint.h>

[[clang::loader_uninitialized]]
__gpu_local uint32_t shared_mem[64];

int factorial (unsigned long cur, int iter, int fin) {
	if (iter < fin) {
		return factorial(cur*iter, iter + 1, fin);
	} else {
		return cur;
	}
}

extern "C" __gpu_kernel void localmem_static_wait(uint32_t *out) {
  shared_mem[__gpu_thread_id(0)] = 2;

  __gpu_sync_threads();

    unsigned long res = 0;
	for (unsigned long i = 0; i < 1000000000; i++) { //< 1000000000; i++) {
		res += factorial(1, 1, 30);
        if (res == 0) {
            res = 1;
        }
	}


  if (__gpu_thread_id(0) == 0) {
    out[__gpu_block_id(0)] = res; //0;
    for (uint32_t i = 0; i < __gpu_num_threads(0); i++)
      out[__gpu_block_id(0)] += shared_mem[i];
  }
}
