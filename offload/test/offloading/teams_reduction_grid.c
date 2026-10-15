// clang-format off
// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: env LIBOMPTARGET_INFO=16 OMP_NUM_TEAMS=2 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=CAP

// REQUIRES: amdgpu
// clang-format on

// The number of blocks of kernels with a cross-team reduction: one block per
// compute unit for a large reduction, and for a scalar reduction as many as
// fill 16 waves per compute unit, but at least one (one block of 1024 threads
// has at least 16 waves). The number of compute units is taken from the
// default grid of a kernel without a reduction, four blocks per compute unit.
// No more blocks than there are iterations, the num_teams clause is honored,
// and OMP_NUM_TEAMS caps the number of blocks.

#include <stdio.h>

#define N (1 << 20)
#define M 8

int A[N];

int main() {
#pragma omp target teams distribute parallel for thread_limit(256)             \
    map(tofrom : A)
  for (int I = 0; I < N; ++I)
    A[I] = 1;
  // CHECK: Launching kernel {{.*}} with [[[#BLK4:]],1,1] blocks and [256,1,1] threads
  // CAP: Launching kernel {{.*}} with [2,1,1] blocks and [256,1,1] threads

  double Sums[M] = {0};
#pragma omp target teams distribute parallel for thread_limit(256)             \
    reduction(+ : Sums[0 : M])
  for (int I = 0; I < N; ++I)
    Sums[I % M] += 1;
  // CHECK: Launching kernel {{.*}} with [[[#div(BLK4,4)]],1,1] blocks and [256,1,1] threads
  // CAP: Launching kernel {{.*}} with [2,1,1] blocks and [256,1,1] threads

  double Sum1 = 0;
#pragma omp target teams distribute parallel for thread_limit(1024)            \
    reduction(+ : Sum1)
  for (int I = 0; I < N; ++I)
    Sum1 += 1;
  // CHECK: Launching kernel {{.*}} with [[[#div(BLK4,4)]],1,1] blocks and [1024,1,1] threads
  // CAP: Launching kernel {{.*}} with [2,1,1] blocks and [1024,1,1] threads

  double Sum2 = 0;
#pragma omp target teams distribute parallel for reduction(+ : Sum2)
  for (int I = 0; I < 16; ++I)
    Sum2 += 1;
  // CHECK: Launching kernel {{.*}} with [1,1,1] blocks

  double Sum3 = 0;
#pragma omp target teams distribute parallel for num_teams(3)                  \
    thread_limit(256) reduction(+ : Sum3)
  for (int I = 0; I < N; ++I)
    Sum3 += 1;
  // CHECK: Launching kernel {{.*}} with [3,1,1] blocks and [256,1,1] threads

  // CHECK: Sums[0]=131072 Sum1=1048576 Sum2=16 Sum3=1048576
  // CAP: Sums[0]=131072 Sum1=1048576 Sum2=16 Sum3=1048576
  printf("Sums[0]=%.0f Sum1=%.0f Sum2=%.0f Sum3=%.0f\n", Sums[0], Sum1, Sum2,
         Sum3);
  return 0;
}
