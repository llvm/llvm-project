// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=NOGRAPH
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=GRAPH

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// The program mapping, between replays, what a taskgraph keeps to itself.  A
// is unmapped when the graph is finalized, so the graph gives it storage of its
// own on each replay; then a `target enter data` maps it between replays.  The
// graph holds nothing between replays, so the mapping is not one it can object
// to, and the replay that finds it borrows it, just as the graph's `target`
// region would share it outside a graph: the graph's own copies of A are off
// for that replay, the mapping's `to` and `from` standing in for them.  The
// replay after the `target exit data` is back on storage of the graph's own.
//
// None of this declines a replay or warns: B, which the graph keeps to itself
// throughout, is not affected either.

#include <stdio.h>

#define N 256
#define ITERS 5

static int A[N], B[N];

int main(void) {
  for (int It = 0; It < ITERS; ++It) {
    if (It == 2) {
#pragma omp target enter data map(to : A[0 : N])
    }
    if (It == 3) {
#pragma omp target exit data map(from : A[0 : N])
    }
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : A[0 : N], B[0 : N])
      for (int I = 0; I < N; ++I) {
        A[I] += 1;
        B[I] += 2;
      }
    }
  }

  int Errors = 0;
  for (int I = 0; I < N; ++I)
    Errors += A[I] != ITERS || B[I] != 2 * ITERS;

  // NOGRAPH-NOT: taskgraph
  // GRAPH-NOT: taskgraph

  // NOGRAPH: taskgraph claim enter data errors: 0
  // GRAPH: taskgraph claim enter data errors: 0
  printf("taskgraph claim enter data errors: %d\n", Errors);
  return Errors != 0;
}
