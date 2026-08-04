// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compileopt-generic
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// REQUIRES: amdgpu

// A cross-team reduction launches with twice the threads per team and half the
// teams of an equivalent non-reduction loop, keeping the total thread count of
// the default grid.

#include <stdio.h>
#include <stdlib.h>

// ceil(SMALL_N / (2 * default workgroup size)) = 3 teams, far below the
// half-default team cap on any GPU.
#define SMALL_N (3 * 512)

int main(void) {
  // High trip count so both grids hit the default team-count cap.
  const int N = 4 * 1024 * 1024;
  int *a = (int *)malloc(sizeof(int) * N);
  for (int i = 0; i < N; ++i)
    a[i] = 1;

  // Reference: capture the default grid
  // CHECK: Launching kernel {{.*}} with {{\[}}[[#REFB:]],1,1] blocks and
  // CHECK-SAME: {{\[}}[[#REFT:]],1,1] threads in SPMD mode
#pragma omp target teams distribute parallel for map(tofrom : a[0 : N])
  for (int i = 0; i < N; ++i)
    a[i] += 1;

  // High trip count: half teams, double threads
  // CHECK: Launching kernel {{.*}} with {{\[}}[[#div(REFB,2)]],1,1] blocks and
  // CHECK-SAME: {{\[}}[[#mul(REFT,2)]],1,1] threads in SPMD mode
  long long sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum)            \
    map(tofrom : a[0 : N])
  for (int i = 0; i < N; ++i)
    sum += a[i];

  // Small trip count: the trip-count bound wins, so teams follow the trip count
  // while threads stay widened.
  // CHECK: Launching kernel {{.*}} with [3,1,1] blocks and
  // CHECK-SAME: {{\[}}[[#mul(REFT,2)]],1,1] threads in SPMD mode
  long long small_sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : small_sum)      \
    map(tofrom : a[0 : SMALL_N])
  for (int i = 0; i < SMALL_N; ++i)
    small_sum += a[i];

  // CHECK: sum = 8388608
  // CHECK: small_sum = 3072
  printf("sum = %lld\n", sum);
  printf("small_sum = %lld\n", small_sum);

  free(a);
  return 0;
}
