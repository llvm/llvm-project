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

// Two things that pull in opposite directions, in one region.
//
// `map(A[Lo : LEN])` names a section at a position that is not a constant, so
// the recorded slot is a host address with no compile-time relationship to the
// variable it came from.  Re-deriving it is a runtime delta: where the variable
// is now, plus the distance the recording had between the variable and the
// section.  A replay from a different frame is what makes the difference
// visible, so the region is reached through a deeper call chain each time.
//
// `firstprivate(saved : Coef)` beside it must *not* move.  Its slot points at a
// snapshot inside the recorded blob -- that is what `saved` means -- so
// relocating it back to the live variable is precisely the thing to avoid,
// while the map next to it still has to be relocated.  Coef is refilled per
// encounter so a replay that read the live copy would be caught by value.
//
// Values outside the window are checked too: an offset re-derived a row out
// would write over them and still leave the window looking right.

#include <stdio.h>

#define N 512
#define LEN 128
#define M 8

// Not a constant to the compiler, and the same on every encounter: what varies
// between replays is the frame, not the section.
static int Lo;

static void *Seen[3];
static int Encounter;

static int work(int Seed) {
  int A[N];
  int Coef[M];
  for (int I = 0; I < N; ++I)
    A[I] = Seed + I;
  for (int J = 0; J < M; ++J)
    Coef[J] = Seed + J;

  Seen[Encounter++] = A;

#pragma omp taskgraph
  {
#pragma omp target firstprivate(saved : Coef) map(tofrom : A[Lo : LEN])
    for (int I = 0; I < LEN; ++I)
      A[Lo + I] += Coef[I % M];
  }

  int Errors = 0;
  for (int I = 0; I < N; ++I) {
    int Want = Seed + I;
    // The first encounter is the one that recorded, so its Coef is the one
    // every replay privatizes from.
    if (I >= Lo && I < Lo + LEN)
      Want += 100 + ((I - Lo) % M);
    if (A[I] != Want)
      ++Errors;
  }
  return Errors;
}

// Reach work() through a deeper chain each time, so its locals -- and with them
// both the variable the section is measured from and the frame a stale slot
// would write into -- sit somewhere else on every encounter.  The padding is
// written and read so it cannot be elided.
static int atDepth(int Depth, int Seed) {
  char Pad[4096];
  Pad[0] = (char)Depth;
  int Errors = Depth ? atDepth(Depth - 1, Seed) : work(Seed);
  return Errors + (Pad[0] != (char)Depth);
}

int main(int argc, char **argv) {
  Lo = 64 + (argc - 1);

  int Errors = 0;
  Errors += atDepth(0, 100);
  Errors += atDepth(3, 200);
  Errors += atDepth(7, 300);

  // CHECK: distinct frames: 3
  int Distinct =
      1 + (Seen[1] != Seen[0]) + (Seen[2] != Seen[0] && Seen[2] != Seen[1]);
  printf("distinct frames: %d\n", Distinct);

  // CHECK: taskgraph dynamic bound with saved firstprivate errors: 0
  printf("taskgraph dynamic bound with saved firstprivate errors: %d\n",
         Errors);

  // Right answers alone would not show the section was re-derived rather than
  // the frames happening to line up, so the count is pinned: one recording
  // encounter and two replays that each relocate.
  // RELOC-COUNT-2: relocated host captures, re-resolving ranges
  // RELOC-NOT: relocated host captures
  return Errors != 0;
}
