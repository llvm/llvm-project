// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_TASKGRAPH_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// The data plan of a graph that maps its data with enter and exit data around a
// target region, as dumped before each replay.
//
// The enter data creates the mappings, so its maps perform the copies in and
// the exit data's perform the copies out; the target region's implicit maps
// fall inside them and are internal, their copies off.  Mapping the struct
// together with the section lets the enter data attach its pointer.  The last
// replay finds the global array mapped outside the graph, which turns its
// copies off.
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
#pragma omp target enter data map(to : Mystruct, Mystruct.Q[0 : 50],           \
                                      Arr[0 : 100])
#pragma omp target firstprivate(Point)
    for (int J = 4; J < 50; ++J)
      Mystruct.Q[J] = Arr[J] + J + K + Point.X + Point.Y;
#pragma omp target exit data map(from : Mystruct, Mystruct.Q[0 : 50],          \
                                     Arr[0 : 100])
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
  // CHECK: sequential {
  // CHECK-NEXT: enter data {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 0/0: mapped (range: [[S]], offset: 0, size: 8, in (perimeter, enabled))
  // CHECK-NEXT: 1/1: mapped (range: [[Q]], offset: 0, size: 200, in (perimeter, enabled))
  // CHECK-NEXT: 2/2: attach (ptr-range: [[S]], offset: 0, size: 8, ptee-range: [[Q]], ptee-offset: 0, staging-offset: 0, enabled)
  // CHECK-NEXT: 3/3: mapped (range: [[ARR]], offset: 0, size: 400, in (perimeter, enabled))
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: copy-in (arg: 0)
  // CHECK-NEXT: 1: copy-in (arg: 1)
  // CHECK-NEXT: 2: copy-in (arg: 3)
  // CHECK-NEXT: 3: attach (arg: 2)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 4/0: mapped (range: [[S]], offset: 0, size: 8, in (internal, disabled), out (internal, disabled))
  // CHECK-NEXT: 5/1: mapped (range: [[ARR]], offset: 0, size: 400, in (internal, disabled), out (internal, disabled))
  // CHECK-NEXT: 6/2: literal (5)
  // CHECK-NEXT: 7/3: firstprivate (base-offset: 0, size: 8, staging-offset: 0, init-pending)
  // CHECK-NEXT: 8/4: literal (0)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: copy-in (arg: 0)
  // CHECK-NEXT: 1: copy-in (arg: 1)
  // CHECK-NEXT: 2: private-init (arg: 3)
  // CHECK-NEXT: 3: launch
  // CHECK-NEXT: 4: copy-out (arg: 0)
  // CHECK-NEXT: 5: copy-out (arg: 1)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: exit data {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 9/0: mapped (range: [[S]], offset: 0, size: 8, out (perimeter, enabled))
  // CHECK-NEXT: 10/1: mapped (range: [[Q]], offset: 0, size: 200, out (perimeter, enabled))
  // CHECK-NEXT: 11/2: mapped (range: [[ARR]], offset: 0, size: 400, out (perimeter, enabled))
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: copy-out (arg: 0)
  // CHECK-NEXT: 1: copy-out (arg: 1)
  // CHECK-NEXT: 2: copy-out (arg: 2)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: Updates:
  // CHECK-NEXT: Arg 0 changes: device-addr
  // CHECK-NEXT: Arg 1 changes: host-addr device-addr
  // CHECK-NEXT: Arg 2 changes: device-addr data enable
  // CHECK-NEXT: Arg 3 changes: device-addr
  // CHECK-NEXT: Arg 4 changes: device-addr enable
  // CHECK-NEXT: Arg 5 changes: device-addr enable
  // CHECK-NEXT: Arg 7 changes: data
  // CHECK-NEXT: Arg 9 changes: device-addr
  // CHECK-NEXT: Arg 10 changes: host-addr device-addr
  // CHECK-NEXT: Arg 11 changes: device-addr
  //
  // The section moved again, and the array is now borrowed.
  // CHECK-LABEL: Target graph pre-replay:
  // CHECK: 3/3: mapped (range: [[ARR]], offset: 0, size: 400, in (perimeter, disabled))
  // CHECK: 11/2: mapped (range: [[ARR]], offset: 0, size: 400, out (perimeter, disabled))
  // CHECK: Updates:
  // CHECK-NEXT: Arg 1 changes: host-addr
  // CHECK-NEXT: Arg 3 changes: enable
  // CHECK-NEXT: Arg 10 changes: host-addr
  // CHECK-NEXT: Arg 11 changes: enable
  // CHECK-NOT: Target graph pre-replay:
  //
  // Each run computes 10 + 10 + 5 + 7 + 9.
  // CHECK: Sum = 123
  printf("Sum = %d\n", Sum);
  return 0;
}
