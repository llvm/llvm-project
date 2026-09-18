// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// map(s.p[0:n]) is the shape that needs pointer attachment.  The struct is
// copied to the device with the host value of s.p still in it, so before a
// kernel can follow that field it has to be overwritten with the device
// address of the pointee.  The runtime does that with a one-word transfer from
// a staging buffer; a graph turns it into an ordinary memcpy node, which is
// only correct if the staging buffer is stable for the life of the graph and
// holds the right address at every replay.
//
// Two things move independently here, and the attachment has to survive both:
// the struct, because the graph is replayed from a different frame, and the
// pointee, because it is reallocated between replays.  A dropped attachment
// leaves a host address in the device struct, so it faults rather than
// answering wrongly; a stale one reaches the previous replay's buffer.

#include <stdio.h>
#include <stdlib.h>

#define N 512

// Tag is never read on the device: map(Local.P[0:N]) maps the bounding box of
// the mapped members, which is P alone, so the rest of the struct is not there.
struct S {
  int *P;
  int Tag;
};

static int *Data;

static int work(int Seed) {
  struct S Local;
  Local.P = Data;
  Local.Tag = Seed;

  for (int I = 0; I < N; ++I)
    Data[I] = Seed + I;

  int Sum = 0;

#pragma omp taskgraph
  {
#pragma omp target map(tofrom : Local.P[0 : N]) map(tofrom : Sum)
    {
      Sum = 0;
      for (int I = 0; I < N; ++I) {
        Local.P[I] += 1;
        Sum += Local.P[I];
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

// Reach work() through a deeper chain each time so its struct sits at a
// different address on every encounter of the one taskgraph directive in it.
static int atDepth(int Depth, int Seed) {
  char Pad[4096];
  Pad[0] = (char)Depth;
  int Errors = Depth ? atDepth(Depth - 1, Seed) : work(Seed);
  return Errors + (Pad[0] != (char)Depth);
}

// Give the pointee a new address, so the pointer the graph recorded is stale
// even though the struct holding it is not.
static int moved(int Seed) {
  int *Old = Data;
  Data = (int *)malloc(N * sizeof(int));
  int Errors = work(Seed);
  free(Old);
  return Errors;
}

int main(void) {
  Data = (int *)malloc(N * sizeof(int));

  int Errors = 0;
  Errors += atDepth(0, 1);
  Errors += atDepth(2, 3);
  Errors += atDepth(5, 7);
  Errors += moved(11);
  Errors += moved(13);

  free(Data);

  // CHECK: taskgraph map attach errors: 0
  printf("taskgraph map attach errors: %d\n", Errors);
  return Errors != 0;
}
