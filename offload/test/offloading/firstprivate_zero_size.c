// RUN: %libomptarget-compile-generic
//
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG
//
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefix=CHECK
//
// REQUIRES: libomptarget-debug

// Test that firstprivate objects of size 0 (which the runtime receives as
// PRIVATE | TO entries with size 0, e.g., a zero-length array or an empty
// struct) do not assert in the runtime and are passed as null.

#include <stdio.h>

struct Empty {};

int main(void) {
  int Zero[0];
  struct Empty E;
  int R = 0;

  // DEBUG: has size 0 - pushing null target argument
  // DEBUG: has size 0 - pushing null target argument
#pragma omp target firstprivate(Zero, E) map(from : R)
  {
    R = 42;
  }

  // CHECK: R = 42
  printf("R = %d\n", R);

  return 0;
}
