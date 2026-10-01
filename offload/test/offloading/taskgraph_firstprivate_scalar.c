// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=CHECK
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefixes=CHECK,GRAPH

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A firstprivate scalar is passed to the kernel by value.  When its variable is
// in the taskgraph construct's enclosing data environment, each replay takes
// the variable's value at that replay (OpenMP 6.0 [14.3]); a `saved` one keeps
// the value it had when the graph was recorded.
//
// The replays run from different stack depths, and one of them is declined to
// the host.  The replay after the decline passes the same value as the
// declined one: the declined replay has already brought the recorded arguments
// up to date, so the value is right there only if the device remembers that it
// has not taken the change in yet.  The last replay repeats a value too, with
// no decline in between.

#include <stdio.h>
#include <string.h>

// The region using N maps nothing, writing its results to a declare target
// variable instead, and the decline is set off through Trigger, which a
// different region maps.  So nothing about the declined replay touches the
// region using N: a replay that re-reports every mapping leaves it alone, and
// its kernel arguments are rewritten because N changed or not at all.
#pragma omp begin declare target
int Out[3];
#pragma omp end declare target
struct Holder {
  int *P;
};
static int Count;
static struct Holder Trigger = {&Count};

static void run(int N, int S, double D) {
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : Trigger.P[0 : 1])
    ++Trigger.P[0];
#pragma omp target firstprivate(N, D) firstprivate(saved : S)
    {
      Out[0] = N;
      Out[1] = S;
      Out[2] = (int)D;
    }
  }
}

// Replays from Depth frames further down, over stack that has been scribbled
// on first.
static void deeper(int Depth, int N, int S, double D) {
  volatile char Scribble[512];
  memset((char *)Scribble, 0x5a, sizeof(Scribble));
  if (Depth)
    deeper(Depth - 1, N, S, D);
  else
    run(N, S, D);
}

int main(void) {
  const int Declined = 2;
  const int N[] = {1, 2, 3, 3, 4, 4};
  int Errors = 0;
  for (int It = 0; It < 6; ++It) {
    // Mapping Trigger from outside the graph declines this one replay: the
    // graph would borrow that mapping of the pointer, and attaching it to the
    // pointee the graph keeps to itself would leave it pointing at memory
    // freed when the replay ends.
    if (It == Declined) {
#pragma omp target enter data map(to : Trigger)
    }
    Out[0] = Out[1] = Out[2] = -1;
#pragma omp target update to(Out)
    deeper(It, N[It], 50 + It, 10.0 * N[It]);
#pragma omp target update from(Out)
    if (It == Declined) {
#pragma omp target exit data map(release : Trigger)
    }
    if (Out[0] != N[It] || Out[1] != 50 || Out[2] != 10 * N[It]) {
      printf("replay %d: N %d (want %d) S %d (want 50) D %d (want %d)\n", It,
             Out[0], N[It], Out[1], Out[2], 10 * N[It]);
      ++Errors;
    }
  }
  Errors += Count != 6;

  // GRAPH: replays on the host this time
  // GRAPH-NOT: replays on the host this time
  // CHECK-NOT: replay {{[0-9]+}}:
  // CHECK: taskgraph firstprivate scalar errors: 0
  printf("taskgraph firstprivate scalar errors: %d\n", Errors);
  return Errors != 0;
}
