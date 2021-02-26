// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu -fopenmp-version=51
// RUN: %libomptarget-run-fail-aarch64-unknown-linux-gnu 2>&1 \
// RUN: | %fcheck-aarch64-unknown-linux-gnu

// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu -fopenmp-version=51
// RUN: %libomptarget-run-fail-powerpc64-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64-ibm-linux-gnu

// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu -fopenmp-version=51
// RUN: %libomptarget-run-fail-powerpc64le-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64le-ibm-linux-gnu

// RUN: %libomptarget-compile-x86_64-pc-linux-gnu -fopenmp-version=51
// RUN: %libomptarget-run-fail-x86_64-pc-linux-gnu 2>&1 \
// RUN: | %fcheck-x86_64-pc-linux-gnu

#include <stdio.h>

int main() {
  int i;

  // CHECK: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  fprintf(stderr, "addr=%p, size=%ld\n", &i, sizeof i);

  // CHECK-NOT: Libomptarget
  #pragma omp target enter data map(alloc: i)
  #pragma omp target exit data map(present, delete: i)

  // CHECK: i was present
  fprintf(stderr, "i was present\n");

  // CHECK: Libomptarget message: device mapping required by 'present' map type modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory
  #pragma omp target exit data map(present, delete: i)

  // CHECK-NOT: i was present
  fprintf(stderr, "i was present\n");

  return 0;
}
