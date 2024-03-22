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

#pragma omp declare target
int foo(int p) { return p * 2; }
#pragma omp end declare target

#define N 10000

int main() {
  double *a = (double *)malloc(sizeof(double) * (N + 1));
  double p = 12.0;

  // Init a:
  for (int i = 0; i < N + 1; i++) {
    a[i] = 0.0;
  }

  p += a[50];

  // Using "<"
  double sum = 0.0;
  double sum2 = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum, sum2)      \
    map(tofrom : sum, sum2)
  for (int i = 0; i < N; i++) {
    a[i] += p + foo(p);
    sum += a[i];
    sum2 += a[i] + 1;
  }

  // clang-format off
  // CHECK: DEVID:  0 SGN:2 {{.*}} tripcount:10000 rpc:1 md:1 md_LB:0 md_UB:4999
  // CHECK: DEVID:  1 SGN:2 {{.*}} tripcount:10000 rpc:1 md:1 md_LB:5000 md_UB:9999
  // clang-format on

  // CHECK: a[40] = 36
  int index = 40;
  printf("a[%d] = %f\n", index, a[index]);

  // CHECK: SUM = 360000
  printf("SUM = %f\n", sum);

  // CHECK: SUM2 = 370000
  printf("SUM2 = %f\n", sum2);

  // Checking the results are correct:
  bool error = false;
  for (int i = 0; i < N; i++) {
    if (!(a[i] == 36)) {
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

  free(a);
  return 0;
}
