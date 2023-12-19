// clang-format off
// RUN: %libomptarget-compilexx-generic && env LIBOMPTARGET_LAZY_DEVICE_INIT=1 LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// REQUIRES: libomptarget-debug

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO

#include <stdio.h>
#include <stdlib.h>

int main() {
  int *a = (int *)malloc(sizeof(int) * 10);

  // clang-format off
// CHECK: omptarget --> Using lazy device initialization!
// CHECK: omptarget --> Plugin adaptor {{.*}} has index 0, exposes 0 out of {{.*}} devices!
// CHECK: omptarget --> Done registering entries!
// CHECK: omptarget --> Use default device id [[DEVICE_ID:.*]]
// CHECK: omptarget --> Device [[DEVICE_ID]] (local ID 0) has been lazily initialized! (IsInit = 1)
  // clang-format on

#pragma omp target map(from : a[ : 10])
  { a[5] = 4; }

  // CHECK: a[5] = 4

  printf("a[5] = %d\n", a[5]);

  free(a);

  return 0;
}
