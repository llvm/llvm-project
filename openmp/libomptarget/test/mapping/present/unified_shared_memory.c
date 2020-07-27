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

// The runtime considers unified shared memory to be always present.
#pragma omp requires unified_shared_memory

int main() {
  int i;

  // CHECK-NOT: Libomptarget
#pragma omp target data map(alloc: i)
#pragma omp target map(present, alloc: i)
  ;

  // CHECK: i is present
  fprintf(stderr, "i is present\n");

  // CHECK-NOT: Libomptarget
#pragma omp target map(present, alloc: i)
  ;

  // CHECK: is present
  fprintf(stderr, "i is present\n");

  return 0;
}
