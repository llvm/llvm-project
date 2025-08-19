// clang-format off
// RUN: %libomptarget-compile-generic -fopenmp-target-multi-device
// RUN: env HSA_XNACK=1 OMPX_APU_MAPS=1 LIBOMPTARGET_NUM_MULTI_DEVICES=2 LIBOMPTARGET_KERNEL_TRACE=1 \
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

// REQUIRES: multi_device

#include <omp.h>
#include <stdio.h>

#define N 2

int main() {
  double *a = (double *)malloc(sizeof(double) * (N + 1));

  // Init a [0, 0, 0]
  for (int i = 0; i < N + 1; i++) {
    a[i] = 0.0;
  }

// Loop with tripcount 1, should be less than number of devices used
// so this should actually only use 1 device:
#pragma omp target teams distribute parallel for
  for (int i = 1; i < N; i++) {
    a[i] += 1;
  }

  // clang-format off
  // CHECK: DEVID:  0 SGN:7 {{.*}} tripcount:1 rpc:0 md:1 md_LB:0 md_UB:0
  // clang-format on

  // CHECK: a[0] = 0
  // CHECK: a[1] = 1
  // CHECK: a[2] = 0
  printf("a[0] = %f\n", a[0]);
  printf("a[1] = %f\n", a[1]);
  printf("a[2] = %f\n", a[2]);

  free(a);
  return 0;
}
