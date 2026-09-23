// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// Exercises *recording + replay* of target / target-data constructs in a
// taskgraph: a taskgraph body containing 'target enter data', a 'target'
// kernel, a 'target update', and 'target exit data' (chained with depend
// clauses) is captured as four discriminated TARGET regions that participate
// in the clone ring, then re-entered several times.
//
// The first (recording) pass forwards each construct to libomptarget live; the
// processed graph is then handed to libomptarget, and subsequent passes are
// replayed through it (__tgt_taskgraph_replay), re-issuing every recorded data
// op and kernel.  Each pass therefore applies the device transform once, so
// the result accumulates across passes.  This validates that recording, graph
// build, replay (re-issue + clone-ring linkage), and teardown all behave.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 1024;
  const int Reps = 3;
  int *a = (int *)malloc(N * sizeof(int));
  int *ref = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    a[i] = ref[i] = i;

  for (int rep = 0; rep < Reps; ++rep) {
#pragma omp taskgraph
    {
#pragma omp target enter data map(to : a[0 : N]) depend(inout : a[0])
#pragma omp target map(alloc : a[0 : N]) depend(inout : a[0])
      {
        for (int i = 0; i < N; ++i)
          a[i] = a[i] * 2 + 1;
      }
#pragma omp target update from(a[0 : N]) depend(inout : a[0])
#pragma omp target exit data map(release : a[0 : N]) depend(inout : a[0])
    }
  }

  // The device transform a -> 2a+1 is applied once per pass (recording +
  // replays), so the result matches the same transform applied Reps times.
  for (int rep = 0; rep < Reps; ++rep)
    for (int i = 0; i < N; ++i)
      ref[i] = ref[i] * 2 + 1;

  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != ref[i])
      ++errors;

  // CHECK: taskgraph target record errors: 0
  printf("taskgraph target record errors: %d\n", errors);
  free(a);
  free(ref);
  return errors != 0;
}
