// clang-format off
// This test verifies correctness of Xteam Reduction for sum reduction using increment.
// 
// RUN: %libomptarget-compile-generic -fopenmp-target-fast
// RUN: env LIBOMPTARGET_KERNEL_TRACE=1 %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

// clang-format on

#include <omp.h>
#include <stdio.h>

int main() {
  int N = 10;
  int sum = 0;

#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (int j = 0; j < N; j = j + 1)
    sum++;

  printf("sum = %d\n", sum);
  int rc = sum != 10;

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
