// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=CHECK
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefixes=CHECK,GRAPH

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// Fails until a plugin claims taskgraphs, which the AMDGPU backend is the
// first to do.
// XFAIL: *

// A replay starts from the taskgraph construct's enclosing data environment,
// and a variable declared inside the region is not part of it: the code that
// declared and initialised it does not run again.  So a firstprivate of such a
// variable takes its value from the saved data environment (OpenMP 6.0
// [14.3]), as though the clause had said `saved`, rather than from wherever its
// storage happened to be when the graph was recorded.
//
// Each replay runs from a different stack depth, and one of them is declined
// to the host, so neither the recording frame nor the device's copy can stand
// in for the saved value by accident.

#include <stdio.h>
#include <string.h>

#define M 16

struct Big {
  int V[M];
};

struct Holder {
  int *P;
};
static int Out;
static struct Holder H = {&Out};

static void run(int N) {
#pragma omp taskgraph
  {
    struct Big C;
    for (int I = 0; I < M; ++I)
      C.V[I] = N;
#pragma omp target firstprivate(C) map(tofrom : H.P[0 : 1])
    H.P[0] = C.V[M - 1];
  }
}

// Replays from Depth frames further down, over stack that has been scribbled
// on first.
static void deeper(int Depth, int N) {
  volatile char Scribble[512];
  memset((char *)Scribble, 0x5a, sizeof(Scribble));
  if (Depth)
    deeper(Depth - 1, N);
  else
    run(N);
}

int main(void) {
  const int Declined = 2;
  int Errors = 0;
  for (int It = 0; It < 5; ++It) {
    // Mapping H from outside the graph declines this one replay: the graph
    // would borrow that mapping of the pointer, and attaching it to the
    // pointee the graph keeps to itself would leave it pointing at memory
    // freed when the replay ends.
    if (It == Declined) {
#pragma omp target enter data map(to : H)
    }
    Out = -1;
    deeper(It, 100 + It);
    if (It == Declined) {
#pragma omp target exit data map(release : H)
    }
    if (Out != 100) {
      printf("replay %d: C %d (want 100)\n", It, Out);
      ++Errors;
    }
  }

  // GRAPH: replays on the host this time
  // GRAPH-NOT: replays on the host this time
  // CHECK-NOT: replay {{[0-9]+}}:
  // CHECK: taskgraph firstprivate inner errors: 0
  printf("taskgraph firstprivate inner errors: %d\n", Errors);
  return Errors != 0;
}
