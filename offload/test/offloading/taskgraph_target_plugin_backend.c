// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: omp_taskgraph_experimental

// Replay of a graph of target regions with the AMDGPU plugin's graph backend
// enabled, so the graph is claimed at finalize and every replay is issued by the
// plugin from its own lowering instead of by libomp on the host.  The result must
// be identical either way; a plugin that declines this graph is also fine, since
// libomp then replays it, so this covers both paths reaching the same answer
// rather than asserting which one ran.
//
// The shape is a diamond: one kernel produces `a`, two independent kernels
// consume it into `b` and `c`, and a fourth joins them into `d`.  So a backend
// that spreads the graph over concurrent lanes has to both run the middle pair
// concurrently and synchronize across lanes at the fork and the join, and the
// transfers of each buffer have to be ordered against the kernels that produce
// and consume it.
//
// The shape comes from the clauses, and both halves are needed: the depend
// clauses supply the three edges, and `nowait` is what leaves everything else
// unordered.  Dropping `nowait` makes each recorded target an undeferred task,
// which takes an encounter-order edge as well and flattens the diamond into a
// four-node chain -- still the right answer, but with none of the concurrency
// this test exists to exercise.  Dropping the depend clauses instead leaves
// four unordered regions and a wrong one.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 1024;
  const int Iters = 4;
  int *a = (int *)malloc(N * sizeof(int));
  int *b = (int *)malloc(N * sizeof(int));
  int *c = (int *)malloc(N * sizeof(int));
  int *d = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = 0;
    c[i] = 0;
    d[i] = 0;
  }

  for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
    {
#pragma omp target nowait map(tofrom : a[0 : N]) depend(out : a[0 : N])
      {
        for (int i = 0; i < N; ++i)
          a[i] = a[i] + 1;
      }
#pragma omp target nowait map(to : a[0 : N]) map(tofrom : b[0 : N])            \
    depend(in : a[0 : N]) depend(out : b[0 : N])
      {
        for (int i = 0; i < N; ++i)
          b[i] = b[i] + a[i];
      }
#pragma omp target nowait map(to : a[0 : N]) map(tofrom : c[0 : N])            \
    depend(in : a[0 : N]) depend(out : c[0 : N])
      {
        for (int i = 0; i < N; ++i)
          c[i] = c[i] + 2 * a[i];
      }
#pragma omp target nowait map(to : b[0 : N], c[0 : N]) map(from : d[0 : N])    \
    depend(in : b[0 : N], c[0 : N])
      {
        for (int i = 0; i < N; ++i)
          d[i] = b[i] + c[i];
      }
    }
  }

  // Encounter k (1-based) leaves a[i] == i + k, so b[i] sums those, c[i] is
  // twice b[i], and d[i] is three times b[i].
  const int Sum = Iters * (Iters + 1) / 2;
  int errors = 0;
  for (int i = 0; i < N; ++i) {
    const int B = Iters * i + Sum;
    if (a[i] != i + Iters || b[i] != B || c[i] != 2 * B || d[i] != 3 * B)
      ++errors;
  }

  // CHECK: taskgraph plugin backend errors: 0
  printf("taskgraph plugin backend errors: %d\n", errors);
  free(d);
  free(c);
  free(b);
  free(a);
  return errors != 0;
}
