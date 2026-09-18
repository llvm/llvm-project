// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// Privatised arguments want device storage that is not a mapping: nothing else
// in the program has a view of it, so it is absent from the mapping table.
// Outside a graph the runtime allocates it per launch, fills it and frees it
// again; a graph cannot, because the address is written into a kernarg image
// when the graph is lowered.  So the plan allocates once for the graph and
// refills the contents each replay.
//
// Both shapes that reach this are exercised, and neither is a corner case:
//
//   - map(p[0:n]) on a plain pointer.  `p` is predetermined firstprivate on the
//     construct and its private copy is initialised by
//     corresponding-pointer-initialization -- the *device* address of the
//     pointee, which is only known once the pointee's own map is resolved.  A
//     stale copy points the kernel at host memory.
//   - firstprivate(S) on an aggregate, with no `saved` modifier, so the copy is
//     of the variable as it stands at this replay.  Storage reused from the last
//     replay without a refill would silently hand the kernel the previous
//     encounter's value.
//
// Each is replayed from a different frame, which is what makes the refill read
// the live variable rather than the recorded address.

#include <stdio.h>
#include <stdlib.h>

#define N 512

struct Agg {
  int A[4];
  double D;
};

static int *Data;

static int pointerCase(int Seed) {
  int *P = Data;
  for (int I = 0; I < N; ++I)
    P[I] = Seed + I;

  int Sum = 0;
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : P[0 : N]) map(tofrom : Sum)
    {
      Sum = 0;
      for (int I = 0; I < N; ++I) {
        P[I] += 1;
        Sum += P[I];
      }
    }
  }

  int Want = 0;
  for (int I = 0; I < N; ++I)
    Want += Seed + I + 1;

  int Errors = (Sum != Want);
  for (int I = 0; I < N; ++I)
    if (Data[I] != Seed + I + 1)
      ++Errors;
  return Errors;
}

static int aggregateCase(int Seed) {
  struct Agg S;
  for (int I = 0; I < 4; ++I)
    S.A[I] = Seed + I;
  S.D = Seed;

  int Sum = 0;
#pragma omp taskgraph
  {
#pragma omp target firstprivate(S) map(tofrom : Sum)
    {
      Sum = (int)S.D;
      for (int I = 0; I < 4; ++I)
        Sum += S.A[I];
    }
  }

  int Want = Seed;
  for (int I = 0; I < 4; ++I)
    Want += Seed + I;
  return Sum != Want;
}

// Reach the case through a deeper chain each time, so its locals sit at a
// different address on every encounter of the taskgraph directive inside it.
static int atDepth(int Depth, int Seed, int (*Fn)(int)) {
  char Pad[4096];
  Pad[0] = (char)Depth;
  int Errors = Depth ? atDepth(Depth - 1, Seed, Fn) : Fn(Seed);
  return Errors + (Pad[0] != (char)Depth);
}

int main(void) {
  Data = (int *)malloc(N * sizeof(int));

  int Errors = 0;
  Errors += atDepth(0, 1, pointerCase);
  Errors += atDepth(2, 3, pointerCase);
  Errors += atDepth(5, 7, pointerCase);
  Errors += atDepth(0, 10, aggregateCase);
  Errors += atDepth(3, 20, aggregateCase);
  Errors += atDepth(6, 30, aggregateCase);

  free(Data);

  // CHECK: taskgraph map private errors: 0
  printf("taskgraph map private errors: %d\n", Errors);
  return Errors != 0;
}
