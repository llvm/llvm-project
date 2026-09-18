// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env HSA_XNACK=1 LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env HSA_XNACK=1 %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// REQUIRES: unified_shared_memory

// Under unified shared memory a map works on the host allocation unless it
// carries `close`, which asks for a device copy instead.  The difference is
// only visible through a map that does not copy back: a kernel writing through
// a `to`-only map changes the host array when the map works on host memory, and
// leaves it untouched when it had a device copy of its own.  So a graph that
// ignored `close` would still get the `tofrom` arrays right and silently leak
// every write through the `close, to` one.  The host path (plugin backend
// off) must agree with the plugin path.

#include <stdio.h>
#include <stdlib.h>

#pragma omp requires unified_shared_memory

int main() {
  const int N = 1024;
  const int Iters = 4;
  int *CloseTo = (int *)malloc(N * sizeof(int));
  int *CloseToFrom = (int *)malloc(N * sizeof(int));
  int *SharedTo = (int *)malloc(N * sizeof(int));
  int *SharedToFrom = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    CloseTo[i] = CloseToFrom[i] = SharedTo[i] = SharedToFrom[i] = i;

  for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
    {
#pragma omp target map(close, to : CloseTo[0 : N])                             \
    map(close, tofrom : CloseToFrom[0 : N]) map(to : SharedTo[0 : N])          \
    map(tofrom : SharedToFrom[0 : N])
      {
        for (int i = 0; i < N; ++i) {
          CloseTo[i] += 1;
          CloseToFrom[i] += 1;
          SharedTo[i] += 1;
          SharedToFrom[i] += 1;
        }
      }
    }
  }

  // Each encounter maps CloseTo afresh from the unchanged host array and throws
  // the device copy away, so the host never sees an update.  Every other array
  // accumulates one per encounter: CloseToFrom through its copy back, and the
  // two maps without `close` because the kernel wrote the host memory itself.
  int errors = 0;
  for (int i = 0; i < N; ++i) {
    if (CloseTo[i] != i)
      ++errors;
    if (CloseToFrom[i] != i + Iters || SharedTo[i] != i + Iters ||
        SharedToFrom[i] != i + Iters)
      ++errors;
  }

  // CHECK: taskgraph map close errors: 0
  printf("taskgraph map close errors: %d\n", errors);
  free(CloseTo);
  free(CloseToFrom);
  free(SharedTo);
  free(SharedToFrom);
  return errors != 0;
}
