// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env KMP_TASKGRAPH_TRACE=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic \
// RUN:   --check-prefixes=CHECK,TRACE
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: omp_taskgraph_experimental

// OpenMP 6.1 [21.5]: the structured block of a 'target data' is a task of its
// own, ordered between the enter-data and exit-data tasks.  A taskgraph
// records exactly those three.  The body task's region is not part of the
// taskgraph region, so what it encounters is not recorded but runs afresh on
// every replay, and taskgraph restrictions (a taskwait needing 'depend', for
// one) do not apply inside it.
//
// The body runs its region in a taskgroup, so the exit-data task that copies
// the data back waits for nested work the body left running: here a nested
// task that is still writing when the body's own code has finished.

#include <stdio.h>
#include <stdlib.h>

#define N 256
#define ITERS 4

int main() {
  int *A = (int *)malloc(N * sizeof(int));
  for (int I = 0; I < N; ++I)
    A[I] = 0;
  int BodyRuns = 0, NestedRuns = 0, AfterWait = 0, BadWait = 0;

  for (int It = 0; It < ITERS; ++It) {
#pragma omp taskgraph
    {
#pragma omp target data map(tofrom : A[0 : N])
      {
        ++BodyRuns;
        int Seen = -1;
#pragma omp task shared(NestedRuns, Seen)
        {
          ++NestedRuns;
          Seen = It;
        }
#pragma omp taskwait
        if (Seen != It)
          ++BadWait;
        ++AfterWait;

        // Left running when the body's code ends; only the body's taskgroup
        // keeps the exit-data task from copying A back before it is done.
#pragma omp task
        {
#pragma omp target map(present, tofrom : A[0 : N])
          for (int I = 0; I < N; ++I)
            A[I] += 1;
        }
      }
    }
  }

  int Errors = 0;
  for (int I = 0; I < N; ++I)
    if (A[I] != ITERS)
      ++Errors;

  // The recorded graph holds the three parts of the construct and nothing
  // from inside the body.
  // TRACE: Processed taskgraph
  // TRACE-NEXT: sequential {
  // TRACE-NEXT:   target enter data:
  // TRACE-NEXT:   node:
  // TRACE-NEXT:   target exit data:
  // TRACE-NEXT: }
  // The enter-data and exit-data each go to the device as a subgraph of their
  // own, and libomp runs the body between them.  A is mapped by nothing
  // outside the graph, so the nested region's 'present' finds it only because
  // the enter-data's subgraph leaves its mapping in place for the exit-data's
  // to remove, as the two constructs would outside a graph.
  // TRACE: Transmitted taskgraph {{.*}} subgraph
  // TRACE: Transmitted taskgraph {{.*}} subgraph
  // TRACE-NOT: replays on the host

  // CHECK: body runs: 4
  printf("body runs: %d\n", BodyRuns);
  // CHECK: nested task runs: 4
  printf("nested task runs: %d\n", NestedRuns);
  // CHECK: past taskwait: 4, saw stale value: 0
  printf("past taskwait: %d, saw stale value: %d\n", AfterWait, BadWait);
  // CHECK: copy-back errors: 0
  printf("copy-back errors: %d\n", Errors);
  free(A);
  return Errors != 0 || BadWait != 0;
}
