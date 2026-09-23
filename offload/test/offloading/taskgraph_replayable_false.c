// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// replayable(false) on the target family inside an 'omp taskgraph' region.
//
// OpenMP 6.0 [14.3] makes a construct in a taskgraph construct replayable
// "unless otherwise specified by the replayable clause", and a replay execution
// does not execute the region body.  So a target construct that opts out is
// launched by the ordinary libomptarget path on the execution that runs the
// body, and does not appear in the record -- it is absent from every replay,
// while its replayable neighbours run on each one.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define NUM_REPS 4 // one recording execution + three replays

int main() {
  int *a = (int *)malloc(N * sizeof(int));
  int *b = (int *)malloc(N * sizeof(int));
  int *c = (int *)malloc(N * sizeof(int));
  int *d = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    a[i] = b[i] = c[i] = d[i] = i;

  // A replayable target next to one that opts out: the first runs on every
  // execution of the region, the second only on the one that records.
  for (int rep = 0; rep < NUM_REPS; ++rep) {
#pragma omp taskgraph graph_id(1)
    {
#pragma omp target map(tofrom : a[0 : N])
      {
        for (int i = 0; i < N; ++i)
          a[i] += 1;
      }
#pragma omp target map(tofrom : b[0 : N]) replayable(false)
      {
        for (int i = 0; i < N; ++i)
          b[i] += 1;
      }
    }
  }

  // A non-replayable 'target update' reading back a buffer that a replayable
  // target updates on the device.  The enclosing 'target data' maps with 'to',
  // so nothing is copied back when it ends and the host only ever sees the
  // value the update-from fetched -- on the recording execution alone.
#pragma omp target data map(to : c[0 : N])
  {
    for (int rep = 0; rep < NUM_REPS; ++rep) {
#pragma omp taskgraph graph_id(2)
      {
#pragma omp target map(alloc : c[0 : N])
        {
          for (int i = 0; i < N; ++i)
            c[i] += 1;
        }
#pragma omp target update from(c[0 : N]) replayable(false)
      }
    }
  }

  // A whole map / compute / read-back chain that opts out: all four constructs
  // run on the recording execution and the record stays empty, so no replay
  // does anything at all.
  for (int rep = 0; rep < NUM_REPS; ++rep) {
#pragma omp taskgraph graph_id(3)
    {
#pragma omp target enter data map(to : d[0 : N]) replayable(false)
#pragma omp target map(alloc : d[0 : N]) replayable(false)
      {
        for (int i = 0; i < N; ++i)
          d[i] += 5;
      }
#pragma omp target update from(d[0 : N]) replayable(false)
#pragma omp target exit data map(release : d[0 : N]) replayable(false)
    }
  }

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    // Replayable: incremented on every execution of the region.
    if (a[i] != i + NUM_REPS)
      ++errors;
    // Not replayable: encountered once, on the recording execution.
    if (b[i] != i + 1)
      ++errors;
    // Device-side increment on every execution, read back on the first only.
    if (c[i] != i + 1)
      ++errors;
    if (d[i] != i + 5)
      ++errors;
  }

  // CHECK: taskgraph replayable(false) target errors: 0
  printf("taskgraph replayable(false) target errors: %d\n", errors);
  free(a);
  free(b);
  free(c);
  free(d);
  return errors != 0;
}
