// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_INFO=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A `target enter data` and a `target` region reading the same data, with
// nothing ordering the two.  Outside a graph, whichever maps first copies the
// data in, and a map is atomic, so the kernel always sees it.  The data plan
// settles which leaf copies when the graph is built, and the region tree is all
// that orders one leaf against another, so here the kernel could read the data
// mid-copy, or before it.  The graph maps A itself on every replay, so the copy
// always runs, and every replay is declined to the host, where the results are
// right.  The decline is reported once.  taskgraph_unordered_copy_replay.c has
// replays that keep the device graph.
//
// The host changes A before every replay, and the kernel reads the end of it,
// which a copy reaches last.

#include <stdio.h>
#include <stdlib.h>

#define N (1024 * 1024)
#define M 64
#define ITERS 4

int main(void) {
  int *A = (int *)malloc(N * sizeof(int));
  int B[M];
  int DepA, DepB, Errors = 0;

  // CHECK: taskgraph {{[0-9]+}} replays on the host: leaf {{[0-9]+}} copies [{{.*}}) and leaf {{[0-9]+}} uses it, and nothing orders the two
  for (int It = 0; It < ITERS; ++It) {
    for (int I = 0; I < N; ++I)
      A[I] = 1000 * It + I % 7;
#pragma omp taskgraph
    {
#pragma omp target map(to : A[0 : N]) map(from : B[0 : M])                     \
    nowait depend(out : DepB)
      for (int I = 0; I < M; ++I)
        B[I] = A[N - M + I];
#pragma omp target enter data map(to : A[0 : N]) nowait depend(out : DepA)
#pragma omp target exit data map(release : A[0 : N])                           \
    nowait depend(in : DepA, DepB)
    }
#pragma omp taskwait
    for (int I = 0; I < M; ++I)
      Errors += B[I] != 1000 * It + (N - M + I) % 7;
  }

  // CHECK: taskgraph unordered copy decline errors: 0
  printf("taskgraph unordered copy decline errors: %d\n", Errors);
  free(A);
  return Errors != 0;
}
