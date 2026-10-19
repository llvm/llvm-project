// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_TASKGRAPH_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// Fails until a plugin claims taskgraphs, which the AMDGPU backend is the
// first to do.
// XFAIL: *

// The data plan of a graph mapping a section through a struct member pointer,
// as dumped before each replay.
//
// The map attaches the device copy of the struct's pointer to the device copy
// of the section.  Each encounter of the graph points the struct at a different
// live allocation, so every replay finds the section at a new host address.
// The last replay finds the global array mapped outside the graph, which turns
// its copy off.
//
// Range ids follow host addresses, so they are matched by size.

#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int *Q;
} S;

int K = 5;
int Arr[100];

static int run(int *Buf) {
  struct {
    int X, Y;
  } Point = {7, 9};
  S Mystruct = {Buf};

#pragma omp taskgraph
  {
#pragma omp target map(tofrom : Mystruct.Q[4 : 50]) map(to : Arr[0 : 100])     \
    firstprivate(Point)
    for (int J = 4; J < 54; ++J)
      Mystruct.Q[J] = Arr[J] + J + K + Point.X + Point.Y;
  }

  return Mystruct.Q[10];
}

int main(void) {
  int *Bufs[3];
  for (int I = 0; I < 3; ++I)
    Bufs[I] = (int *)calloc(100, sizeof(int));
  for (int I = 0; I < 100; ++I)
    Arr[I] = I;

  int Sum = run(Bufs[0]) + run(Bufs[1]);
#pragma omp target enter data map(to : Arr[0 : 100])
  Sum += run(Bufs[2]);
#pragma omp target exit data map(release : Arr[0 : 100])

  for (int I = 0; I < 3; ++I)
    free(Bufs[I]);

  // CHECK-LABEL: Target graph pre-replay:
  // CHECK: Ranges:
  // CHECK-DAG: Id: [[ARR:[0-9]+]], {{.*}}, Size: 400{{$}}
  // CHECK-DAG: Id: [[Q:[0-9]+]], {{.*}}, Size: 200{{$}}
  // CHECK-DAG: Id: [[S:[0-9]+]], {{.*}}, Size: 8{{$}}
  // CHECK: Firstprivate staging buffer size: 8
  // CHECK: attach staging buffer size: 8
  // CHECK: target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 0/0: mapped (range: [[S]], offset: 0, size: 8, in (perimeter, enabled), out (perimeter, enabled))
  // CHECK-NEXT: 1/1: mapped (range: [[Q]], offset: 0, size: 200, in (perimeter, enabled), out (perimeter, enabled))
  // CHECK-NEXT: 2/2: attach (ptr-range: [[S]], offset: 0, size: 8, ptee-range: [[Q]], ptee-offset: 0, staging-offset: 0, enabled)
  // CHECK-NEXT: 3/3: mapped (range: [[ARR]], offset: 0, size: 400, in (perimeter, enabled))
  // CHECK-NEXT: 4/4: literal (5)
  // CHECK-NEXT: 5/5: firstprivate (base-offset: 0, size: 8, staging-offset: 0, init-pending)
  // CHECK-NEXT: 6/6: literal (0)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: copy-in (arg: 0)
  // CHECK-NEXT: 1: copy-in (arg: 1)
  // CHECK-NEXT: 2: copy-in (arg: 3)
  // CHECK-NEXT: 3: attach (arg: 2)
  // CHECK-NEXT: 4: private-init (arg: 5)
  // CHECK-NEXT: 5: launch
  // CHECK-NEXT: 6: copy-out (arg: 0)
  // CHECK-NEXT: 7: copy-out (arg: 1)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: Updates:
  // CHECK-NEXT: Arg 0 changes: device-addr
  // CHECK-NEXT: Arg 1 changes: host-addr device-addr
  // CHECK-NEXT: Arg 2 changes: device-addr data enable
  // CHECK-NEXT: Arg 3 changes: device-addr
  // CHECK-NEXT: Arg 5 changes: data
  //
  // The section moved again, and the array is now borrowed.
  // CHECK-LABEL: Target graph pre-replay:
  // CHECK: 3/3: mapped (range: [[ARR]], offset: 0, size: 400, in (perimeter, disabled))
  // CHECK: Updates:
  // CHECK-NEXT: Arg 1 changes: host-addr
  // CHECK-NEXT: Arg 3 changes: enable
  // CHECK-NOT: Target graph pre-replay:
  //
  // Each run computes 10 + 10 + 5 + 7 + 9.
  // CHECK: Sum = 123
  printf("Sum = %d\n", Sum);
  return 0;
}
