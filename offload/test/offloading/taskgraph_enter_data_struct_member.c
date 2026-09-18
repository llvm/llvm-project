// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN:   env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic

// REQUIRES: gpu
// REQUIRES: omp_taskgraph_experimental

// A graph that maps struct members with `target enter data` and unmaps them
// with `target exit data`.  Such a map is a struct entry followed by its
// members, and only the struct entry changes the reference count: a member
// copies on whichever change its struct's entry owns.  So every replay copies
// the members in on the enter and out on the exit, and the device never shows
// a replay what the previous one left behind.
//
// The host changes the data between replays, so a missing copy-in shows as the
// previous replay's values.  `e` is released rather than copied back, so it
// must come back unchanged.

#include <stdio.h>

#define N 4
#define ITERS 4

struct S {
  int Pad;
  int D[N];
  int E[N];
};

int main(void) {
  struct S s = {0};
  int Errors = 0;

  for (int It = 0; It < ITERS; ++It) {
    for (int I = 0; I < N; ++I)
      s.D[I] = s.E[I] = 100 * It + I;
#pragma omp taskgraph
    {
#pragma omp target enter data map(to : s.D[0 : N], s.E[0 : N])
#pragma omp target map(present, alloc : s.D[0 : N], s.E[0 : N])
      for (int I = 0; I < N; ++I) {
        s.D[I] += 1;
        s.E[I] += 2;
      }
#pragma omp target exit data map(from : s.D[0 : N]) map(release : s.E[0 : N])
    }
    for (int I = 0; I < N; ++I)
      Errors += s.D[I] != 100 * It + I + 1 || s.E[I] != 100 * It + I;
  }

  // CHECK: taskgraph enter data struct member errors: 0
  printf("taskgraph enter data struct member errors: %d\n", Errors);
  return Errors != 0;
}
