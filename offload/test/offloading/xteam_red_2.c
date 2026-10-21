// clang-format off
// This test verifies the result of a cross-team reduction kernel with the
// default grid size. teams_reduction_grid.c verifies the grid size.
// RUN: %libomptarget-compile-generic -fopenmp-target-fast
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

// clang-format on
#include <stdio.h>

int main() {
  int N = 1000000;

  double a[N];

  for (int i = 0; i < N; i++)
    a[i] = i;

  double sum1;
  sum1 = 0;

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1)
  for (int j = 0; j < N; j = j + 1)
    sum1 += a[j];

  printf("sum1=%f\n", sum1);

  return 0;
}
// clang-format off
/// CHECK: sum1=499999500000.000000

