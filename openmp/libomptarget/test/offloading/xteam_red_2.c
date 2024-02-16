// clang-format off
// This test verifies that the reduction kernel is of Xteam reduction
// type and is launched with as many teams as the number of CUs.
// RUN: %libomptarget-compile-generic -fopenmp-target-fast
// RUN: env LIBOMPTARGET_DEBUG=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

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
/// CHECK: xteam-red:NumCUs=[[CU_COUNT:[0-9]+]]
/// CHECK: xteam-red:NumGroups=[[CU_COUNT]]

