// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// REQUIRES: unified_shared_memory

#include <stdio.h>

// The runtime considers unified shared memory to be always present.
#pragma omp requires unified_shared_memory

int main() {
  int i;

  // CHECK-NOT: Libomptarget
#pragma omp target data map(alloc : i)
#pragma omp target map(present, alloc : i)
  ;

  // CHECK: i is present
  fprintf(stderr, "i is present\n");

  // CHECK-NOT: Libomptarget
#pragma omp target map(present, alloc : i)
  ;

  // CHECK: is present
  fprintf(stderr, "i is present\n");

  return 0;
}
