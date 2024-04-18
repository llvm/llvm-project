// clang-format off
// This test verifies that the reduction kernel is of Xteam-reduction type
// and is launched with 1920 teams and 8 threads in each team. 
// 
// RUN: %libomptarget-compile-generic -fopenmp-target-fast -fopenmp-target-fast-reduction
// RUN: env LIBOMPTARGET_KERNEL_TRACE=1 LIBOMPTARGET_AMDGPU_LOW_TRIPCOUNT=15360 LIBOMPTARGET_AMDGPU_ADJUST_XTEAM_RED_TEAMS=32 \
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
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: teamsXthrds:(1920X   8)

