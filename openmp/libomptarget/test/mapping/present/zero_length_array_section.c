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
  int arr[5];

  // CHECK: addr=0x[[#%x,HOST_ADDR:]]
  fprintf(stderr, "addr=%p\n", arr);

  // CHECK-NOT: Libomptarget
#pragma omp target data map(alloc: arr[0:5])
#pragma omp target map(present, alloc: arr[0:0])
  ;

  // CHECK: arr is present
  fprintf(stderr, "arr is present\n");

  // arr[0:0] doesn't create an actual mapping in the first directive.
  //
  // CHECK: Libomptarget message: device mapping required by 'present' map type modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] (0 bytes)
  // CHECK: Libomptarget error: Call to getOrAllocTgtPtr returned null pointer ('present' map type modifier).
  // CHECK: Libomptarget error: Call to targetDataBegin failed, abort target.
  // CHECK: Libomptarget error: Failed to process data before launching the kernel.
  // CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory
#pragma omp target data map(alloc: arr[0:0])
#pragma omp target map(present, alloc: arr[0:0])
  ;

  // CHECK-NOT: arr is present
  fprintf(stderr, "arr is present\n");

  return 0;
}
