// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_INFO=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// Two unordered `target` regions each mapping all of T `tofrom` and writing
// half of it.  When the graph maps T itself, each copies T in and out, and
// nothing orders those copies against the other kernel, so the replay goes to
// the host.  When the program has mapped T around the replay, the graph copies
// nothing and keeps the device graph.  The decision is the replay's: the same
// graph goes to the host and comes back as the program maps and unmaps T, and
// says so once each time it starts going to the host.
//
// The markers go to stderr, as the runtime's messages do, so that the two
// interleave in order.

#include <stdio.h>

#define N 4096
#define ITERS 7

int main(void) {
  int T[N];
  int Dep1, Dep2, Errors = 0;
  for (int I = 0; I < N; ++I)
    T[I] = 0;

  for (int It = 0; It < ITERS; ++It) {
    const int Mapped = It < 3 || It == 6;
    // The first encounter records; replays start at 1.
    if (It > 0)
      fprintf(stderr, "replay %d %s\n", It, Mapped ? "mapped" : "unmapped");
    if (Mapped) {
#pragma omp target enter data map(to : T[0 : N])
    }
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : T[0 : N]) nowait depend(out : Dep1)
      for (int I = 0; I < N / 2; ++I)
        T[I] += 1;
#pragma omp target map(tofrom : T[0 : N]) nowait depend(out : Dep2)
      for (int I = N / 2; I < N; ++I)
        T[I] += 2;
    }
#pragma omp taskwait
    if (Mapped) {
#pragma omp target exit data map(from : T[0 : N])
    }
    for (int I = 0; I < N; ++I)
      Errors += T[I] != (I < N / 2 ? 1 : 2) * (It + 1);
  }

  // A replay on the host enters each kernel through the runtime, which says so
  // under LIBOMPTARGET_INFO; one on the device does not.
  //
  // CHECK-NOT: replays on the host
  // CHECK: replay 1 mapped
  // CHECK-NOT: {{Entering OpenMP kernel|replays on the host}}
  // CHECK: replay 2 mapped
  // CHECK-NOT: {{Entering OpenMP kernel|replays on the host}}
  // CHECK: replay 3 unmapped
  // CHECK-NEXT: taskgraph {{[0-9]+}} replays on the host: leaf {{[0-9]+}} copies [{{.*}}) and leaf {{[0-9]+}} uses it, and nothing orders the two
  // CHECK: Entering OpenMP kernel
  // CHECK: replay 4 unmapped
  // CHECK-NOT: replays on the host
  // CHECK: Entering OpenMP kernel
  // CHECK: replay 5 unmapped
  // CHECK-NOT: replays on the host
  // CHECK: Entering OpenMP kernel
  // CHECK: replay 6 mapped
  // CHECK-NOT: {{Entering OpenMP kernel|replays on the host}}
  // CHECK: taskgraph unordered copy replay errors: 0
  fprintf(stderr, "taskgraph unordered copy replay errors: %d\n", Errors);
  return Errors != 0;
}
