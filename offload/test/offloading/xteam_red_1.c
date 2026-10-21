// clang-format off
// This test verifies the result of a fast cross-team reduction kernel under a
// low trip count. A fast reduction has no reduction data, so its grid is that
// of a plain SPMD kernel, which depends on the device.
//
// Cross-team reductions are emitted as plain SPMD kernels (SGN:2) since the
// downstream Xteam reduction execution mode (SGN:8) was removed.
//
// RUN: %libomptarget-compile-generic -fopenmp-target-fast -fopenmp-target-fast-reduction
// RUN: env LIBOMPTARGET_KERNEL_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

// clang-format on
#include <stdio.h>

int main() {
  int N = 15360;

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
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK-SAME: tripcount:15360
/// CHECK: sum1=117957120.000000

