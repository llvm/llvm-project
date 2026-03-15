// clang-format off
// This test verifies correctness of Xteam Reduction when a reference to a reduction
// variable is passed to a function. Currently, Xteam reduction kicks in for a subset
// of these cases.
// 
// RUN: %libomptarget-compile-generic -fopenmp-target-fast
// RUN: env LIBOMPTARGET_KERNEL_TRACE=1 %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

// clang-format on

#include <omp.h>
#include <stdio.h>

int compute_sum_res(int j, double &result, double a[]) {
  result += a[j];
  return 1;
}

void compute_sum(int j, double &result, double a[]) { result += a[j]; }

double compute_sum_rval(int j, double rval, double a[]) { return rval + a[j]; }

int foo(int i) { return 2 * i; }

int main() {
  int N = 10000;

  double a[N];

  for (int i = 0; i < N; i++)
    a[i] = i;

  double sum1, sum2, sum3, sum4, sum5;
  sum1 = sum2 = sum3 = sum4 = sum5 = 0;

  int res = 0;
#pragma omp target teams distribute parallel for reduction(+:sum1) map(tofrom:res)
  for (int j = 0; j < N; j = j + 1)
    res = compute_sum_res(j, sum1, a);

#pragma omp target teams distribute parallel for reduction(+ : sum2)
  for (int j = 0; j < N; j = j + 1)
    compute_sum(j, sum2, a);

#pragma omp target teams distribute parallel for reduction(+ : sum3)
  for (int j = 0; j < N; j = j + 1)
    sum3 = compute_sum_rval(j, sum3, a);

#pragma omp target teams distribute parallel for reduction(+ : sum4)
  for (int j = 0; j < N; j = j + 1)
    foo(compute_sum_res(j, sum4, a));

#pragma omp target teams distribute parallel for reduction(+ : sum5)
  for (int j = 0; j < N; j = j + 1)
    compute_sum_res(j, sum5, a);

  printf("%f %f %f %f %f\n", sum1, sum2, sum3, sum4, sum5);

  int rc = (sum1 != 49995000) || (sum2 != 49995000) || (sum3 != 49995000) ||
           (sum4 != 49995000) || (sum5 != 49995000);

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
