// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A map over a region that is already mapped from outside the graph performs no
// copy: the reference count goes up rather than to one, and the enclosing
// region owns the transfers.  `always` is the clause that overrides that, and a
// graph has to honour it per replay, because the host writes it exists to carry
// happen between replays and the recording cannot know about them.
//
// The two arrays here differ only in the modifier, so B is the control: it
// shows what the borrowed range does without `always` -- no copy in either
// direction, so the host's own writes stay and the doubling never reaches it.
// Asserting that too keeps the test honest, since a version of the runtime that
// copied everything unconditionally would satisfy A's check alone.

#include <stdio.h>
#include <stdlib.h>

#define N 256
#define ITERS 3

int main(void) {
  int *A = (int *)malloc(N * sizeof(int));
  int *B = (int *)malloc(N * sizeof(int));
  for (int I = 0; I < N; ++I)
    A[I] = B[I] = 0;

  int AlwaysErrors = 0, BorrowErrors = 0;

  // Mapped out here, so both maps inside the graph borrow rather than map.
#pragma omp target data map(tofrom : A[0 : N], B[0 : N])
  {
    for (int It = 0; It < ITERS; ++It) {
      for (int I = 0; I < N; ++I) {
        A[I] = 100 * (It + 1) + I;
        B[I] = 100 * (It + 1) + I;
      }

#pragma omp taskgraph
      {
#pragma omp target map(always, tofrom : A[0 : N])
        for (int I = 0; I < N; ++I)
          A[I] *= 2;

#pragma omp target map(tofrom : B[0 : N])
        for (int I = 0; I < N; ++I)
          B[I] *= 2;
      }

      for (int I = 0; I < N; ++I) {
        if (A[I] != 2 * (100 * (It + 1) + I))
          ++AlwaysErrors;
        if (B[I] != 100 * (It + 1) + I)
          ++BorrowErrors;
      }
    }
  }

  // CHECK: always over a borrowed range errors: 0
  printf("always over a borrowed range errors: %d\n", AlwaysErrors);
  // CHECK: borrowed without always, copies performed: 0
  printf("borrowed without always, copies performed: %d\n", BorrowErrors);

  free(B);
  free(A);
  return (AlwaysErrors + BorrowErrors) != 0;
}
