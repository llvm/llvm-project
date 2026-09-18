// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=NOGRAPH
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=GRAPH

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// What a taskgraph keeps to itself is claimed only while the subgraph holding
// it runs.  The graph maps a stack array, replayed from frames at different
// depths, so relocation moves the graph's range with the array -- and once the
// target construct has run, the claim has to be gone from whichever frame the
// replay used.  A host task ordered after the construct runs once the subgraph
// is over, so asking about the array there finds it unmapped and unclaimed, as
// it would be outside a graph, and nothing warns.

#include <omp.h>
#include <stdio.h>

#define N 256

static int Dev, Hits, Errors;

static void work(void) {
  int Local[N];
  for (int I = 0; I < N; ++I)
    Local[I] = I;

#pragma omp taskgraph
  {
#pragma omp target map(tofrom : Local[0 : N]) depend(inout : Local[0])
    for (int I = 0; I < N; ++I)
      Local[I] += 1;
#pragma omp task shared(Local) depend(in : Local[0])
    Hits += omp_target_is_present(Local, Dev);
  }

  for (int I = 0; I < N; ++I)
    Errors += Local[I] != I + 1;
}

static void atDepth(int Depth) {
  volatile char Pad[4096];
  Pad[0] = (char)Depth;
  if (Depth)
    atDepth(Depth - 1);
  else
    work();
  Errors += Pad[0] != (char)Depth;
}

int main(void) {
  Dev = omp_get_default_device();
  atDepth(0);
  atDepth(2);
  atDepth(2);
  atDepth(5);

  // NOGRAPH-NOT: asked about
  // GRAPH-NOT: asked about

  // NOGRAPH: hits 0 errors 0
  // GRAPH: hits 0 errors 0
  printf("hits %d errors %d\n", Hits, Errors);
  return Errors != 0;
}
