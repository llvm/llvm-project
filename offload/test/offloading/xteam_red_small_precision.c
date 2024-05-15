// clang-format off
// This test verifies correctness of Xteam Reduction for reduced precision types.
// 
// RUN: %libomptarget-compile-generic -fopenmp-target-fast -lmlir_float16_utils
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
  int N = 100;

  _Float16 a[N];
  __bf16 b[N];
  short c[N];

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i;
    c[i] = i;
  }

  _Float16 sum1 = 0;
  __bf16 sum2 = 0;
  short sum3 = 0;

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1)
  for (int j = 0; j < N; j = j + 1)
    sum1 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum2) reduction(+:sum2)
  for (int j = 0; j < N; j = j + 2)
    sum2 += b[j];

#pragma omp target teams distribute parallel for map(tofrom:sum3) reduction(+:sum3)
  for (int j = 0; j < N; j = j + 2)
    sum3 += c[j];

  printf("%f %f %d\n", (float)sum1, (float)sum2, sum3);

  int rc = (sum1 != 4952) || (sum2 != 2448) || (sum3 != 2450);

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
