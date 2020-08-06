// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu -fopenmp-version=51
// RUN: %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 \
// RUN: | %fcheck-aarch64-unknown-linux-gnu

// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu -fopenmp-version=51
// RUN: %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64-ibm-linux-gnu

// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu -fopenmp-version=51
// RUN: %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64le-ibm-linux-gnu

// RUN: %libomptarget-compile-x86_64-pc-linux-gnu -fopenmp-version=51
// RUN: %libomptarget-run-x86_64-pc-linux-gnu 2>&1 \
// RUN: | %fcheck-x86_64-pc-linux-gnu

#include <stdio.h>

int main() {
  int i;

#pragma omp target enter data map(alloc:i)

  // i isn't present at the end of the target data region, but the "present"
  // modifier is only checked at the beginning of a region.
#pragma omp target data map(present, alloc: i)
  {
#pragma omp target exit data map(delete:i)
  }

  // CHECK-NOT: Libomptarget
  // CHECK: success
  // CHECK-NOT: Libomptarget
  fprintf(stderr, "success\n");

  return 0;
}
