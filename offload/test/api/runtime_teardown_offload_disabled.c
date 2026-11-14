// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled LIBOMPTARGET_DEBUG=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// Offload disabled: no shutdown callback, teardown must still happen.

// REQUIRES: libomptarget-debug

#include <stdio.h>

int main() {
  int X = 0;
#pragma omp target map(tofrom : X)
  X = 1;
  printf("X = %d\n", X);
  return X != 1;
}

// CHECK-DAG: Offload is disabled. Skipping plugin initialization
// CHECK-DAG: X = 1
// CHECK-DAG: Tearing down offload library!
// CHECK-NOT: Tearing down offload library!
