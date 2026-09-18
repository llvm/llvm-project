// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=NOGRAPH
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=GRAPH

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// Two taskgraphs keeping overlapping memory to themselves.  Neither is mapped
// outside its own replays, so each replay has its own storage and they take
// turns without either seeing the other's -- exactly as two `target` regions
// mapping the overlapping sections one after the other do outside a graph.
//
// A graph has its memory only while it replays, and each replay here is over
// before the next graph starts, so nothing the program does between them is
// warned about: not the second graph's recording encounter, which maps its
// section like any host code, and not the lookups once both are done.

#include <omp.h>
#include <stdio.h>

#define N 256
#define ITERS 3

static int A[N + N / 2];

int main(void) {
  int Dev = omp_get_default_device();
  for (int It = 0; It < ITERS; ++It) {
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : A[0 : N])
      for (int I = 0; I < N; ++I)
        A[I] += 1;
    }
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : A[N / 2 : N])
      for (int I = N / 2; I < N + N / 2; ++I)
        A[I] += 2;
    }
  }

  int Hits = 0;
  Hits += omp_target_is_present(&A[0], Dev);
  Hits += omp_target_is_present(&A[1], Dev);
  Hits += omp_target_is_present(&A[N / 2], Dev);
  Hits += omp_target_is_present(&A[N], Dev);

  int Errors = 0;
  for (int I = 0; I < N + N / 2; ++I)
    Errors += A[I] != (I < N ? ITERS : 0) + (I >= N / 2 ? 2 * ITERS : 0);

  // NOGRAPH-NOT: taskgraph
  // GRAPH-NOT: taskgraph
  // GRAPH-NOT: asked about

  // NOGRAPH: hits 0 errors 0
  // GRAPH: hits 0 errors 0
  printf("hits %d errors %d\n", Hits, Errors);
  return Errors != 0;
}
