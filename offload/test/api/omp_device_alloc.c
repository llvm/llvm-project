// RUN: %libomptarget-compile-run-and-check-generic

// UNSUPPORTED: nvidiagpu
// UNSUPPORTED: amdgpu

#include <assert.h>
#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp target
  {
    int *ptr;
#pragma omp allocate(ptr) allocator(omp_default_mem_alloc)
    ptr = omp_alloc(sizeof(int), omp_default_mem_alloc);
    assert(ptr && "Ptr is (null)!");
    *ptr = 0;
#pragma omp parallel num_threads(32)
    {
#pragma omp atomic
      *ptr += 1;
    }
    assert(*ptr == 32 && "Ptr is not 32");
    omp_free(ptr, omp_default_mem_alloc);
  }

  // CHECK: PASS
  printf("PASS\n");
}
