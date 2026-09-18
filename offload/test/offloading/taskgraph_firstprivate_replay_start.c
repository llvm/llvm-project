// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --implicit-check-not='replays on the host'

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A replay re-encounters every recorded construct before any of its tasks
// runs (OpenMP 6.0 [14.3]), so a target region's by-value firstprivate takes
// the value its variable has when the replay starts, not the value a task it
// depends on has written by the time it is launched.  Both ways of issuing
// the region have to agree on this: in the first run the plugin claims no
// graph and libomp issues the region itself, and in the second the region is
// replayed as a device subgraph, after the writer, a host task libomp runs.
//
// Only the replays are checked: when the graph is recorded, the writer may
// run before the target region is encountered, and then the region sees the
// write.

#include <stdio.h>

#pragma omp begin declare target
int Out;
#pragma omp end declare target

static void run(int N) {
#pragma omp taskgraph
  {
#pragma omp task shared(N) depend(out : N)
    N = -1;
#pragma omp target firstprivate(N) depend(in : N)
    Out = N;
  }
}

int main(void) {
  int Errors = 0;
  for (int It = 1; It <= 4; ++It) {
    Out = -2;
#pragma omp target update to(Out)
    run(It);
#pragma omp target update from(Out)
    if (It > 1 && Out != It) {
      printf("replay %d: N %d (want %d)\n", It, Out, It);
      ++Errors;
    }
  }

  // CHECK-NOT: replay {{[0-9]+}}:
  // CHECK: taskgraph firstprivate replay start errors: 0
  printf("taskgraph firstprivate replay start errors: %d\n", Errors);
  return Errors != 0;
}
