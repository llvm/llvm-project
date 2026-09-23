// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env KMP_TASKGRAPH_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic \
// RUN:     --implicit-check-not='exclusive'
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: env KMP_TASKGRAPH_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic \
// RUN:     --implicit-check-not='exclusive'

// REQUIRES: amdgpu
// REQUIRES: taskgraph

// The other half of taskgraph_target_mutex_exclusive.c: a `mutexinoutset` on a
// target that the region builder cannot express structurally, so the mutex sets
// survive recording and the graph goes back to the host.
//
//        T1 (out: a)        T2 (out: b)
//         |        \        /        |
//         |         \      /         |
//      TA{m0}        TAB{m0,m1}      TB{m1}
//         |               |          |
//         '------ T6 (in: x, y, z) --'
//
// The three mutex targets have different predecessor sets (a, b, and both), so
// the merge pass cannot fold them into one EXCLUSIVE region the way it does for
// the twins in the exclusive test.  Those same differing predecessors make the
// shape non-series-parallel, so the members land in an IRREDUCIBLE region --
// where __kmp_taskgraph_strip_mutex_sets recurses without setting in_exclusive
// and the sets are kept, since nothing else is left to carry the exclusion.
//
// A surviving set is what the AMDGPU lowering declines on, its schedule having
// no way to express "unordered but not simultaneous", so the graph is handed
// back and libomp replays it on the host, taking the recorded locks around each
// target as it issues it.  Declining is the designed outcome here rather than a
// missing feature; a backend that grew a way to serialize the members of an
// irreducible region could stop declining, and would then need this test
// updated to match.
//
// So the run pins all three: the irreducible shape with no exclusive formed, the
// sets surviving into replay, and the decline -- plus the answer, which the
// ordering the mutex targets do have (after T1/T2, before T6) still has to make
// come out right, so the exclusion cannot be "fixed" by serializing the graph
// and losing its dependence structure.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 256;
  const int Iters = 3;
  int Da, Db, Dm0, Dm1, Dx, Dy, Dz;
  int *a = (int *)malloc(N * sizeof(int));
  int *b = (int *)malloc(N * sizeof(int));
  int *x = (int *)malloc(N * sizeof(int));
  int *y = (int *)malloc(N * sizeof(int));
  int *z = (int *)malloc(N * sizeof(int));
  int *r = (int *)malloc(N * sizeof(int));

  int errors = 0;
  for (int it = 0; it < Iters; ++it) {
    for (int i = 0; i < N; ++i)
      a[i] = b[i] = x[i] = y[i] = z[i] = r[i] = -1;

#pragma omp taskgraph
    {
#pragma omp target nowait map(from : a[0 : N]) depend(out : Da)
      {
        for (int i = 0; i < N; ++i)
          a[i] = i + 1;
      }
#pragma omp target nowait map(from : b[0 : N]) depend(out : Db)
      {
        for (int i = 0; i < N; ++i)
          b[i] = 2 * (i + 1);
      }
#pragma omp target nowait map(to : a[0 : N]) map(from : x[0 : N])              \
    depend(in : Da) depend(mutexinoutset : Dm0) depend(out : Dx)
      {
        for (int i = 0; i < N; ++i)
          x[i] = a[i];
      }
#pragma omp target nowait map(to : a[0 : N], b[0 : N]) map(from : z[0 : N])    \
    depend(in : Da, Db) depend(mutexinoutset : Dm0, Dm1) depend(out : Dz)
      {
        for (int i = 0; i < N; ++i)
          z[i] = a[i] + b[i];
      }
#pragma omp target nowait map(to : b[0 : N]) map(from : y[0 : N])              \
    depend(in : Db) depend(mutexinoutset : Dm1) depend(out : Dy)
      {
        for (int i = 0; i < N; ++i)
          y[i] = b[i];
      }
#pragma omp target nowait map(to : x[0 : N], y[0 : N], z[0 : N])               \
    map(from : r[0 : N]) depend(in : Dx, Dy, Dz)
      {
        for (int i = 0; i < N; ++i)
          r[i] = x[i] + y[i] + z[i];
      }
    }

    // x = i+1, y = 2(i+1), z = 3(i+1), so r = 6(i+1).
    for (int i = 0; i < N; ++i)
      if (x[i] != i + 1 || y[i] != 2 * (i + 1) || z[i] != 3 * (i + 1) ||
          r[i] != 6 * (i + 1))
        ++errors;
  }

  // CHECK: irreducible {
  // CHECK: sets:
  // CHECK: declined by the device plugin

  // CHECK: taskgraph target mutex irreducible errors: 0
  printf("taskgraph target mutex irreducible errors: %d\n", errors);
  free(r);
  free(z);
  free(y);
  free(x);
  free(b);
  free(a);
  return errors != 0;
}
