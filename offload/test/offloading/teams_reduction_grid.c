// clang-format off
// This test verifies the grid size of kernels with a cross-team reduction:
// one block per compute unit for a large reduction, the given number of waves
// per compute unit for a scalar one, no more blocks than there are iterations,
// the grid of a non-reduction kernel for a loop too short for one block per
// compute unit, and the num_teams clause honored. The number of compute units
// is taken from the first kernel.
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
#include <omp.h>
#include <stdio.h>

#define N (1 << 20)
#define M 8

int main() {
  double Sum = 0;
  double Sums[M] = {0};
  int NumCUs = 0;

  // Large reduction (64 bytes per team): one block per compute unit.
#pragma omp target teams distribute parallel for thread_limit(256)             \
    reduction(+ : Sums[0 : M]) map(from : NumCUs)
  for (int I = 0; I < N; ++I) {
    Sums[I % M] += 1;
    if (I == 0)
      NumCUs = omp_get_num_teams();
  }
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

  // No more blocks than there are iterations: two blocks of 256 threads per
  // compute unit.
  Sum = 0;
#pragma omp target teams distribute parallel for thread_limit(256)             \
    reduction(+ : Sum)
  for (int I = 0; I < NumCUs * 512; ++I)
    Sum += 1;
  // CHECK: SGN:2
  // CHECK-SAME: teamsXthrds:({{ *}}[[#mul(CUS,2)]]X 256)
  double Sum2 = Sum;

  // Too short for one block of the default size per compute unit: the grid of
  // a non-reduction kernel. As there are more default blocks than compute
  // units, that is one block of 32 threads per compute unit.
  Sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : Sum)
  for (int I = 0; I < NumCUs * 32; ++I)
    Sum += 1;
  // CHECK: SGN:2
  // CHECK-SAME: teamsXthrds:({{ *}}[[#CUS]]X 32)
  double Sum3 = Sum;

  // The num_teams clause is honored.
  Sum = 0;
#pragma omp target teams distribute parallel for num_teams(3)                  \
    thread_limit(256) reduction(+ : Sum)
  for (int I = 0; I < N; ++I)
    Sum += 1;
  // CHECK: SGN:2
  // CHECK-SAME: teamsXthrds:({{ *}}3X 256)

  // The kernel trace goes to stderr, so check the results after all of it.
  // CHECK: Sums[0]=131072 Sum1=1048576 Sum2=[[#mul(CUS,512)]]
  // CHECK-SAME: Sum3=[[#mul(CUS,32)]] Sum4=1048576
  printf("Sums[0]=%.0f Sum1=%.0f Sum2=%.0f Sum3=%.0f Sum4=%.0f\n", Sums[0],
         Sum1, Sum2, Sum3, Sum);
  return 0;
}
