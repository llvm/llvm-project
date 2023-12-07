// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

#include <stdio.h>

int main() {
  int arr[5];

  // CHECK: addr=0x[[#%x,HOST_ADDR:]]
  fprintf(stderr, "addr=%p\n", arr);

  // CHECK-NOT: Libomptarget
#pragma omp target data map(alloc : arr[0 : 5])
#pragma omp target map(present, alloc : arr[0 : 0])
  ;

  // CHECK: arr is present
  fprintf(stderr, "arr is present\n");

  // arr[0:0] doesn't create an actual mapping in the first directive.
  //
  // CHECK: Libomptarget message: device mapping required by 'present' map type modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] (0 bytes)
  // CHECK: Libomptarget error: Call to getTargetPointer returned null pointer ('present' map type modifier).
  // CHECK: Libomptarget error: Call to targetDataBegin failed, abort target.
  // CHECK: Libomptarget error: Failed to process data before launching the kernel.
  // CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory
#pragma omp target data map(alloc : arr[0 : 0])
#pragma omp target map(present, alloc : arr[0 : 0])
  ;

  // CHECK-NOT: arr is present
  fprintf(stderr, "arr is present\n");

  return 0;
}
