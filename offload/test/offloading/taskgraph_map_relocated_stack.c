// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_TASKGRAPH_REPORT_DRIFT=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=RELOC

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// Fails until a plugin claims taskgraphs, which the AMDGPU backend is the
// first to do.
// XFAIL: *

// A recorded target region keeps the map arrays it was launched with, so the
// host addresses in them are the ones the mapped variables had on the
// encounter that recorded the graph.  Nothing says a later encounter runs from
// the same frame: the same function reached through a different call chain
// puts its locals somewhere else entirely, and replaying against the recorded
// addresses would then read and write a dead frame.
//
// clang emits a relocation callback beside the recorded region, the runtime
// calls it with the taskgraph record reconstructed for this encounter, and it
// rewrites the map arrays; libomptarget then follows its ranges to wherever
// they landed.  Without any part of that the graph writes its result into the
// recording frame and the array checked here is left untouched, so the value
// check below is the real test -- the address report only shows that the three
// encounters genuinely ran from different frames.

#include <stdio.h>

#define N 1024

static void *Seen[3];
static int Encounter;

static int work(int Seed) {
  int A[N];
  for (int I = 0; I < N; ++I)
    A[I] = Seed + I;

  Seen[Encounter++] = A;

#pragma omp taskgraph
  {
#pragma omp target map(tofrom : A[0 : N])
    for (int I = 0; I < N; ++I)
      A[I] += 1;
  }

  int Errors = 0;
  for (int I = 0; I < N; ++I)
    if (A[I] != Seed + I + 1)
      ++Errors;
  return Errors;
}

// Reach work() through a deeper chain each time, so its locals sit at a
// different address on every encounter of the one taskgraph directive inside
// it.  The padding is written and read so it cannot be elided.
static int atDepth(int Depth, int Seed) {
  char Pad[4096];
  Pad[0] = (char)Depth;
  int Errors = Depth ? atDepth(Depth - 1, Seed) : work(Seed);
  return Errors + (Pad[0] != (char)Depth);
}

int main(void) {
  int Errors = 0;
  Errors += atDepth(0, 100);
  Errors += atDepth(3, 200);
  Errors += atDepth(7, 300);

  // CHECK: distinct frames: 3
  int Distinct =
      1 + (Seen[1] != Seen[0]) + (Seen[2] != Seen[0] && Seen[2] != Seen[1]);
  printf("distinct frames: %d\n", Distinct);

  // CHECK: taskgraph relocated stack errors: 0
  printf("taskgraph relocated stack errors: %d\n", Errors);

  // Right answers alone would not show relocation was taken rather than the
  // frames happening to line up, so the count is pinned: the first encounter
  // records and the two after it each replay from a frame of their own.
  // RELOC-COUNT-2: relocated host captures, re-resolving ranges
  // RELOC-NOT: relocated host captures
  return Errors != 0;
}
