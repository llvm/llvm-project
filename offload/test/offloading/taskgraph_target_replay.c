// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// Re-encounters an 'omp taskgraph' that contains a 'target' region several
// times.  The first encounter records the graph and launches live; because the
// graph contains a target, the processed region tree is handed to libomptarget
// (__tgt_taskgraph_start/.../_finalize) and every subsequent encounter is
// replayed through libomptarget (__tgt_taskgraph_replay) instead of the host
// exec_descr path.  All encounters must produce the same per-iteration effect.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 1024;
  const int Iters = 4;
  int *a = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    a[i] = i;

  for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : a[0 : N])
      {
        for (int i = 0; i < N; ++i)
          a[i] = a[i] + 1;
      }
    }
  }

  // Each of the Iters encounters adds 1, so a[i] == i + Iters.
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != i + Iters)
      ++errors;

  // CHECK: taskgraph target replay errors: 0
  printf("taskgraph target replay errors: %d\n", errors);
  free(a);
  return errors != 0;
}
