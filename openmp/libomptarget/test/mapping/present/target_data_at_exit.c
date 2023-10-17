// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

#include <stdio.h>

int main() {
  int i;

#pragma omp target enter data map(alloc : i)

  // i isn't present at the end of the target data region, but the "present"
  // modifier is only checked at the beginning of a region.
#pragma omp target data map(present, alloc : i)
  {
#pragma omp target exit data map(delete : i)
  }

  // CHECK-NOT: Libomptarget
  // CHECK: success
  // CHECK-NOT: Libomptarget
  fprintf(stderr, "success\n");

  return 0;
}
