// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN:   | %fcheck-generic

// The runtime is torn down exactly once at exit.

// REQUIRES: libomptarget-debug

#include <stdio.h>

int main() {
  int X = 0;
#pragma omp target map(tofrom : X)
  X = 1;
  printf("X = %d\n", X);
  return X != 1;
}

// CHECK-DAG: X = 1
// CHECK-DAG: Tearing down offload library!
// CHECK-NOT: Tearing down offload library!
