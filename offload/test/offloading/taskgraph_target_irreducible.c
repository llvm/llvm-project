// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// An NxN lattice "wavefront" of 'target' kernels inside an 'omp taskgraph',
// re-encountered several times.  Cell (i,j) depends on (i-1,j) and (i,j-1) --
// the canonical non-series-parallel (irreducible) dependence graph.  The region
// builder carves it into a single irreducible region and streams its explicit
// intra-kernel edges to libomptarget; on replay the device dispatches the
// kernels honoring those edges (plugin-owned graph) or the software fallback
// re-issues them in the carve's topological emission order.  Either way the
// wavefront values (binomial C(i+j, i)) must come out right every iteration.
//
// `nowait` is what keeps the wavefront edges the only ones.  A recorded target
// without it generates an undeferred task, so encounter order is a dependence
// too, and the extra edges make the graph a total order -- which is trivially
// series-parallel, so the carve would come out sequential and the irreducible
// path this test is named for would never run.

#include <stdio.h>
#include <stdlib.h>

#define N 4
#define ITERS 4

int main() {
  long long *v = (long long *)malloc(N * N * sizeof(long long));
  long long ref[N][N];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ref[i][j] =
          (i == 0 && j == 0)
              ? 1
              : ((i > 0 ? ref[i - 1][j] : 0) + (j > 0 ? ref[i][j - 1] : 0));

  int errors = 0;
  for (int it = 0; it < ITERS; ++it) {
    for (int k = 0; k < N * N; ++k)
      v[k] = -1;
#pragma omp taskgraph
    {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
          int me = i * N + j;
          int up = (i - 1) * N + j;
          int left = i * N + (j - 1);
          // clang-format off
          if (i == 0 && j == 0) {
#pragma omp target nowait map(tofrom : v[0 : N * N]) depend(out : v[me])
            { v[me] = 1; }
          } else if (i == 0) {
#pragma omp target nowait map(tofrom : v[0 : N * N]) \
    depend(in : v[left]) depend(out : v[me])
            { v[me] = v[left]; }
          } else if (j == 0) {
#pragma omp target nowait map(tofrom : v[0 : N * N]) \
    depend(in : v[up]) depend(out : v[me])
            { v[me] = v[up]; }
          } else {
#pragma omp target nowait map(tofrom : v[0 : N * N]) \
    depend(in : v[up], v[left]) depend(out : v[me])
            { v[me] = v[up] + v[left]; }
          }
          // clang-format on
        }
    }
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        if (v[i * N + j] != ref[i][j])
          ++errors;
  }

  // CHECK: taskgraph irreducible target errors: 0
  printf("taskgraph irreducible target errors: %d\n", errors);
  free(v);
  return errors != 0;
}
