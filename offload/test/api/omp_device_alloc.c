// RUN: %libomptarget-compile-run-and-check-generic

#include <assert.h>
#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp target teams num_teams(4)
#pragma omp parallel
  {
    int *ptr = (int *)omp_alloc(sizeof(int), omp_default_mem_alloc);
    assert(ptr && "Ptr is (null)!");
    *ptr = 1;
    assert(*ptr == 1 && "Ptr is not 1");
    omp_free(ptr, omp_default_mem_alloc);
  }

#pragma omp target
  {
    assert(!omp_alloc(sizeof(int), omp_null_allocator) && "Ptr is not (null)!");
  }

  // CHECK: PASS
  printf("PASS\n");
}
