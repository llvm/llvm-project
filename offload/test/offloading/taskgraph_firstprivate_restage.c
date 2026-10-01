// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=CHECK
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefixes=CHECK,GRAPH

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A graph copies a firstprivate aggregate into its private storage only when
// the value it is initialised from has changed since the device last saw it,
// since nothing on the device writes that storage.  So the value has to be
// tracked against what the device holds, not against what was last staged.
//
// B is an ordinary firstprivate of a variable the taskgraph construct's
// enclosing data environment has, so each replay sees it as it stands then: it
// repeats a value, returns to an earlier one, and changes on a replay that is
// declined to the host, then keeps that value on the replay after, which the
// device has not seen yet.  S is `saved`, so every replay sees the value it had
// when the graph was recorded, however the variable has changed since.

#include <stdio.h>

#define M 16

struct Big {
  int V[M];
};

struct Holder {
  int *P;
};
static int Out[2];
static struct Holder H = {Out};

static void run(struct Big B, struct Big S) {
#pragma omp taskgraph
  {
#pragma omp target firstprivate(B) firstprivate(saved : S)                     \
    map(tofrom : H.P[0 : 2])
    {
      H.P[0] = B.V[M - 1];
      H.P[1] = S.V[M - 1];
    }
  }
}

static struct Big fill(int X) {
  struct Big R;
  for (int I = 0; I < M; ++I)
    R.V[I] = X;
  return R;
}

int main(void) {
  const int Values[] = {1, 2, 2, 1, 3, 3, 3, 4};
  const int Declined = 4;
  int Errors = 0;
  for (int It = 0; It < (int)(sizeof(Values) / sizeof(Values[0])); ++It) {
    // Mapping H from outside the graph declines this one replay: the graph
    // would borrow that mapping of the pointer, and attaching it to the
    // pointee the graph keeps to itself would leave it pointing at memory
    // freed when the replay ends.
    if (It == Declined) {
#pragma omp target enter data map(to : H)
    }
    run(fill(Values[It]), fill(100 + It));
    if (It == Declined) {
#pragma omp target exit data map(release : H)
    }
    if (Out[0] != Values[It] || Out[1] != 100) {
      printf("replay %d: B %d (want %d), S %d (want 100)\n", It, Out[0],
             Values[It], Out[1]);
      ++Errors;
    }
  }

  // GRAPH: replays on the host this time
  // GRAPH-NOT: replays on the host this time
  // CHECK-NOT: replay {{[0-9]+}}:
  // CHECK: taskgraph firstprivate restage errors: 0
  printf("taskgraph firstprivate restage errors: %d\n", Errors);
  return Errors != 0;
}
