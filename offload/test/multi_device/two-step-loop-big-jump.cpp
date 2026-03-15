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

#define N 10000

int main() {
  double *a = (double *)malloc(sizeof(double) * (N + 1));

  // Init a:
  for (int i = 0; i < N + 1; i++) {
    a[i] = 0.0;
  }

// Using "<"
#pragma omp target teams
#pragma omp distribute parallel for
  for (int i = 7; i < N; i++) {
    a[i] += 1;
  }

  int UB = N;
#pragma omp target teams
#pragma omp distribute parallel for
  for (int i = 7; i < UB; i++) {
    a[i] += 2;
  }

  int LB = 7;
#pragma omp target teams
#pragma omp distribute parallel for
  for (int i = LB; i < UB; i++) {
    a[i] += 3;
  }

// Using "<="
#pragma omp target teams
#pragma omp distribute parallel for
  for (int i = 7; i <= N - 1; i++) {
    a[i] += 1;
  }

#pragma omp target teams
#pragma omp distribute parallel for
  for (int i = 7; i <= UB - 1; i++) {
    a[i] += 2;
  }

#pragma omp target teams
#pragma omp distribute parallel for
  for (int i = LB; i <= UB - 1; i++) {
    a[i] += 3;
  }

  // clang-format off
  // CHECK: DEVID:  0 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:0 md_UB:4995
  // CHECK: DEVID:  1 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:4996 md_UB:9992

  // CHECK: DEVID:  0 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:0 md_UB:4995
  // CHECK: DEVID:  1 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:4996 md_UB:9992

  // CHECK: DEVID:  0 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:0 md_UB:4995
  // CHECK: DEVID:  1 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:4996 md_UB:9992

  // CHECK: DEVID:  0 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:0 md_UB:4995
  // CHECK: DEVID:  1 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:4996 md_UB:9992

  // CHECK: DEVID:  0 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:0 md_UB:4995
  // CHECK: DEVID:  1 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:4996 md_UB:9992

  // CHECK: DEVID:  0 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:0 md_UB:4995
  // CHECK: DEVID:  1 SGN:7 {{.*}} tripcount:9993 rpc:0 md:1 md_LB:4996 md_UB:9992
  // clang-format on

  // CHECK: a[40] = 12
  int index = 40;
  printf("a[%d] = %f\n", index, a[index]);

  // Checking the results are correct:
  bool error = false;
  for (int i = 7; i < N; i++) {
    if (!(a[i] == 12)) {
      printf("ERROR at index = %d, value is a[%d] = %f\n", i, i, a[i]);
      error = true;
      break;
    }
  }

  // CHECK: SUCCESS
  if (!error)
    printf("SUCCESS\n");

  error = false;
  if (a[N] != 0)
    error = true;

  // CHECK: SUCCESS: last entry
  if (!error)
    printf("SUCCESS: last entry\n");

  error = false;
  for (int i = 0; i < 7; i++) {
    if (!(a[i] == 0)) {
      printf("ERROR at index = %d, value is a[%d] = %f\n", i, i, a[i]);
      error = true;
      break;
    }
  }

  // CHECK: SUCCESS: first 7 entries
  if (!error)
    printf("SUCCESS: first 7 entries\n");

  free(a);
  return 0;
}
