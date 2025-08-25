// clang-format off
// RUN: %libomptarget-compile-generic -fopenmp-target-multi-device
// RUN: env HSA_XNACK=1 OMPX_APU_MAPS=1 LIBOMPTARGET_KERNEL_TRACE=1 LIBOMPTARGET_NUM_MULTI_DEVICES=2 \
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

#define N 2000
#define M 5

int main() {
  double *a = (double *)malloc(sizeof(double) * (N * M + 1));

  // Init a:
  for (int i = 0; i < N * M + 1; i++) {
    a[i] = 0.0;
  }

#pragma omp target teams distribute parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      a[i * M + j] += 1;
    }
  }

  // clang-format off
  // CHECK: DEVID:  0 SGN:7 {{.*}} tripcount:10000 rpc:0 md:1 md_LB:0 md_UB:4999
  // CHECK: DEVID:  1 SGN:7 {{.*}} tripcount:10000 rpc:0 md:1 md_LB:5000 md_UB:9999
  // clang-format on

  // Checking the results are correct:
  bool error = false;
  for (int i = 0; i < N * M; i++) {
    if (!(a[i] == 1)) {
      printf("ERROR at index = %d, value is a[%d] = %f\n", i, i, a[i]);
      error = true;
      break;
    }
  }

  // CHECK: SUCCESS
  if (!error)
    printf("SUCCESS\n");

  error = false;
  if (a[N * M] != 0)
    error = true;

  // CHECK: SUCCESS: last entry
  if (!error)
    printf("SUCCESS: last entry\n");

  free(a);
  return 0;
}
