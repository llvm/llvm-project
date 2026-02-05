// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

#include <stdio.h>

int main() {
  // CHECK: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  int x = 111;
  fprintf(stderr, "addr=%p, size=%ld\n", &x, sizeof(x));

  // clang-format off
  // CHECK: omptarget message: device mapping required by 'present' map type modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK: omptarget error: Pointer 0x{{0*}}[[#HOST_ADDR]] was not present on the device upon entry to the region.
  // CHECK: omptarget error: Call to targetDataBegin failed, abort target.
  // CHECK: omptarget error: Failed to process data before launching the kernel.
  // CHECK: omptarget fatal error 1: failure of target construct while offloading is mandatory
  // clang-format on
#pragma omp target map(alloc : x) map(present, alloc : x) map(tofrom : x)
  {
    printf("%d\n", x);
  }

  return 0;
}
