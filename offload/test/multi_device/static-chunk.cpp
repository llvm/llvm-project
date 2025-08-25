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

#define N 10000

int main() {
  double *a = (double *)malloc(sizeof(double) * (N + 1));

  for (int i = 0; i < N + 1; i++) {
    a[i] = 0.0;
  }

// Runs in multi-device mode
#pragma omp target teams distribute parallel for schedule(static, 1)
  for (int i = 1; i < N; i++) {
    a[i] += 2;
  }

// Does not run in multi-device node yet
#pragma omp target teams distribute parallel for schedule(static, 2)
  for (int i = 1; i < N; i++) {
    a[i] += 2;
  }

  // clang-format off
  // CHECK: DEVID:  0 SGN:7 {{.*}} tripcount:9999 rpc:1 md:1 md_LB:0 md_UB:4998
  // CHECK: DEVID:  1 SGN:7 {{.*}} tripcount:9999 rpc:1 md:1 md_LB:4999 md_UB:9998

  // CHECK: DEVID:  0 SGN:2 {{.*}} tripcount:9999 rpc:1 md:0
  // clang-format on

  // CHECK: a[0] = 0
  // CHECK: a[10000] = 0
  printf("a[0] = %f\n", a[0]);
  printf("a[10000] = %f\n", a[10000]);

  bool error = false;
  for (int i = 1; i < N; i++) {
    if (a[i] != 4) {
      error = true;
      printf("ERROR: a[%d] = %f\n", i, a[i]);
      break;
    }
  }

  // CHECK: SUCCESS
  if (!error)
    printf("SUCCESS\n");

  free(a);
  return 0;
}
