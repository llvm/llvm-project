// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %not --crash env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %not --crash %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefixes=CHECK,PLUGIN

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// `present` says the region is expected to be mapped already and this map must
// not bring it in.  A graph cannot settle that at record time: whether a region
// is mapped belongs to the program's state, so the encounter that recorded may
// find it there and a replay may not.  Getting it wrong is silent -- the graph
// would allocate a buffer of its own, and then two device copies of one host
// range exist and the program gets plausible wrong answers -- so the required
// behaviour is a runtime error, per replay, on whichever path is running.
//
// The mapping is therefore made only for the recording encounter and deleted
// after it, so the graph is recorded against a present range and replayed
// against an absent one.  Both paths report and abort: the host path re-runs
// the ordinary map machinery and gets the usual diagnostic, while the plugin
// path fails resolving the range before it submits anything, says so, and
// declines the replay to the host path, which then aborts the same way.

#include <stdio.h>
#include <stdlib.h>

#define N 256

int main(void) {
  int *A = (int *)malloc(N * sizeof(int));
  for (int I = 0; I < N; ++I)
    A[I] = I;

  for (int It = 0; It < 2; ++It) {
    if (It == 0) {
#pragma omp target enter data map(to : A[0 : N])
    }

    // CHECK: encounter 0
    // CHECK: encounter 1
    printf("encounter %d\n", It);
    fflush(stdout);

#pragma omp taskgraph
    {
#pragma omp target map(present, tofrom : A[0 : N])
      for (int I = 0; I < N; ++I)
        A[I] += 1;
    }

    if (It == 0) {
#pragma omp target exit data map(delete : A[0 : N])
    }
  }

  // PLUGIN: taskgraph requires {{.*}} to be present, and it is not
  // PLUGIN: taskgraph {{[0-9]+}} replays on the host this time
  // CHECK: 'present' map type modifier does not exist

  // CHECK-NOT: reached the end
  printf("reached the end\n");
  free(A);
  return 0;
}
