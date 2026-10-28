// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: env LIBOMPTARGET_INFO=16 OMP_NUM_TEAMS=8 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefix=FEW
// RUN: env LIBOMPTARGET_INFO=16 OMP_NUM_TEAMS=1000000 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefix=MANY

// REQUIRES: amdgpu
// clang-format on

// The number of blocks and threads of kernels in Generic-SPMD mode: by default
// six blocks per compute unit, then threads traded for blocks down to one
// wavefront while the distribute loop has an iteration for every block. A
// thread_limit clause turns the trade off. OMP_NUM_TEAMS caps the number of
// blocks, also for the trade. The number of compute units is taken from the
// default grid of an SPMD kernel, four blocks per compute unit.

#include <stdio.h>

#define N (1 << 22)
#define M 64

int A[N];

int main() {
#pragma omp target teams distribute parallel for thread_limit(256)             \
    map(tofrom : A)
  for (int I = 0; I < N; ++I)
    A[I] = 1;
  // CHECK: Launching kernel {{.*}} with [[[#BLK4:]],1,1] blocks and [256,1,1] threads in SPMD mode

#pragma omp target teams distribute thread_limit(256) map(tofrom : A)
  for (int I = 0; I < N / M; ++I) {
#pragma omp parallel for
    for (int J = 0; J < M; ++J)
      A[I * M + J] += 1;
  }
  // CHECK: Launching kernel {{.*}} with [[[#div(mul(BLK4,3),2)]],1,1] blocks and [256,1,1] threads in Generic-SPMD mode

#pragma omp target teams distribute map(tofrom : A)
  for (int I = 0; I < N / M; ++I) {
#pragma omp parallel for
    for (int J = 0; J < M; ++J)
      A[I * M + J] += 1;
  }
  // CHECK: Launching kernel {{.*}} with [{{[0-9]+}},1,1] blocks and [{{32|64}},1,1] threads in Generic-SPMD mode
  // FEW: Launching kernel {{.*}} with [8,1,1] blocks and [256,1,1] threads in Generic-SPMD mode
  // FEW: Launching kernel {{.*}} with [8,1,1] blocks and [256,1,1] threads in Generic-SPMD mode
  // MANY: Launching kernel {{.*}} with [{{[0-9]+}},1,1] blocks and [{{32|64}},1,1] threads in Generic-SPMD mode

#pragma omp target teams distribute map(tofrom : A)
  for (int I = 0; I < 10; ++I) {
#pragma omp parallel for
    for (int J = 0; J < M; ++J)
      A[I * M + J] += 1;
  }
  // CHECK: Launching kernel {{.*}} with [10,1,1] blocks and [{{[0-9]+}},1,1] threads in Generic-SPMD mode

  // CHECK: A[0]=4 A[N-1]=3
  printf("A[0]=%d A[N-1]=%d\n", A[0], A[N - 1]);
  return 0;
}
