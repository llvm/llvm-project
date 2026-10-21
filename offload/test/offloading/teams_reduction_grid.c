// clang-format off
// This test verifies the grid size of kernels with a cross-team reduction:
// one block per compute unit for a large reduction, the given number of waves
// per compute unit for a scalar one, no more blocks than there are iterations,
// and the num_teams clause honored. The number of compute units is taken from
// the first kernel.
//
// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_KERNEL_TRACE=1 LIBOMPTARGET_TEAMS_REDUCTION_WAVES_PER_CU=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

// clang-format on
#include <stdio.h>

#define N (1 << 20)
#define M 8

int main() {
  double Sum = 0;
  double Sums[M] = {0};

  // Large reduction (64 bytes per team): one block per compute unit.
#pragma omp target teams distribute parallel for thread_limit(256)             \
    reduction(+ : Sums[0 : M])
  for (int I = 0; I < N; ++I)
    Sums[I % M] += 1;
  // CHECK: SGN:2
  // CHECK-SAME: teamsXthrds:({{ *}}[[#CUS:]]X 256)

  // Scalar reduction: 16 waves, that is four blocks of 256 threads, per
  // compute unit.
#pragma omp target teams distribute parallel for thread_limit(256)             \
    reduction(+ : Sum)
  for (int I = 0; I < N; ++I)
    Sum += 1;
  // CHECK: SGN:2
  // CHECK-SAME: teamsXthrds:({{ *}}[[#mul(CUS,4)]]X 256)
  double Sum1 = Sum;

  // No more blocks than there are iterations.
  Sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : Sum)
  for (int I = 0; I < 16; ++I)
    Sum += 1;
  // CHECK: SGN:2
  // CHECK-SAME: teamsXthrds:({{ *}}1X
  double Sum2 = Sum;

  // The num_teams clause is honored.
  Sum = 0;
#pragma omp target teams distribute parallel for num_teams(3)                  \
    thread_limit(256) reduction(+ : Sum)
  for (int I = 0; I < N; ++I)
    Sum += 1;
  // CHECK: SGN:2
  // CHECK-SAME: teamsXthrds:({{ *}}3X 256)

  // The kernel trace goes to stderr, so check the results after all of it.
  // CHECK: Sums[0]=131072 Sum1=1048576 Sum2=16 Sum3=1048576
  printf("Sums[0]=%.0f Sum1=%.0f Sum2=%.0f Sum3=%.0f\n", Sums[0], Sum1, Sum2,
         Sum);
  return 0;
}
