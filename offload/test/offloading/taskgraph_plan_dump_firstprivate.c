// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_TASKGRAPH_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// Fails until a plugin claims taskgraphs, which the AMDGPU backend is the
// first to do.
// XFAIL: *

// The data plan of a graph with privatised and by-value arguments, as dumped
// before each replay.
//
//   - An aggregate firstprivate without `saved` is re-staged every replay; one
//     with `saved` is staged once, from the value at recording.
//   - A scalar firstprivate is a literal, re-read every replay, so a change in
//     it is reported as a data update.
//   - Two sections of an array mapped outside the graph merge into one range,
//     and their copies are off, the outside mapping standing in for them.
//
// Range ids follow host addresses, so they are matched by size.  The staged
// aggregates are 16 bytes so that their staging offsets do not depend on the
// alignment of their host addresses.

#include <stdio.h>

struct Quad {
  int X, Y, Z, W;
};

int K = 5;

int main(void) {
  int Arr[100], Arr2[100];
  struct Quad Q = {0, 2, 0, 0};
  for (int I = 0; I < 100; ++I)
    Arr2[I] = I;

#pragma omp target enter data map(to : Arr2[0 : 100])
  for (int It = 0; It < 3; ++It) {
    int Scale = It;
    Q.X = It;
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : Arr[0 : 100]) map(to : Arr2[4 : 50])           \
    firstprivate(saved : K) firstprivate(Q, Scale)
      for (int J = 0; J < 100; ++J)
        Arr[J] = J * Scale + K + Q.X + Q.Y;
#pragma omp target map(tofrom : Arr[0 : 100]) map(from : Arr2[40 : 50])        \
    firstprivate(saved : K, Q)
      for (int J = 0; J < 100; ++J)
        Arr[J] += J + K + Q.X + Q.Y;
    }
  }
#pragma omp target exit data map(from : Arr2[0 : 100])

  // CHECK-LABEL: Target graph pre-replay:
  // CHECK: Ranges:
  // CHECK-DAG: Id: [[ARR2:[0-9]+]], {{.*}}, Size: 344{{$}}
  // CHECK-DAG: Id: [[ARR:[0-9]+]], {{.*}}, Size: 400{{$}}
  // CHECK: Firstprivate staging buffer size: 32
  // CHECK: attach staging buffer size: 0
  // CHECK: sequential {
  // CHECK-NEXT: target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 0/0: mapped (range: [[ARR]], offset: 0, size: 400, in (perimeter, enabled), out (perimeter, enabled))
  // CHECK-NEXT: 1/1: literal (1)
  // CHECK-NEXT: 2/2: literal (5)
  // CHECK-NEXT: 3/3: firstprivate (base-offset: 0, size: 16, staging-offset: 0, init-pending)
  // CHECK-NEXT: 4/4: mapped (range: [[ARR2]], offset: 0, size: 200, in (perimeter, disabled))
  // CHECK-NEXT: 5/5: literal (0)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: copy-in (arg: 0)
  // CHECK-NEXT: 1: copy-in (arg: 4)
  // CHECK-NEXT: 2: private-init (arg: 3)
  // CHECK-NEXT: 3: launch
  // CHECK-NEXT: 4: copy-out (arg: 0)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 6/0: mapped (range: [[ARR]], offset: 0, size: 400, in (perimeter, enabled), out (perimeter, enabled))
  // CHECK-NEXT: 7/1: literal (5)
  // CHECK-NEXT: 8/2: firstprivate (base-offset: 0, size: 16, staging-offset: 16, saved, init-pending)
  // CHECK-NEXT: 9/3: mapped (range: [[ARR2]], offset: 144, size: 200, out (perimeter, disabled))
  // CHECK-NEXT: 10/4: literal (0)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: copy-in (arg: 0)
  // CHECK-NEXT: 1: private-init (arg: 2)
  // CHECK-NEXT: 2: launch
  // CHECK-NEXT: 3: copy-out (arg: 0)
  // CHECK-NEXT: 4: copy-out (arg: 3)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: Updates:
  // CHECK-NEXT: Arg 0 changes: device-addr
  // CHECK-NEXT: Arg 1 changes: data
  // CHECK-NEXT: Arg 3 changes: data
  // CHECK-NEXT: Arg 4 changes: device-addr enable
  // CHECK-NEXT: Arg 6 changes: device-addr
  // CHECK-NEXT: Arg 8 changes: data
  // CHECK-NEXT: Arg 9 changes: device-addr enable
  //
  // The second replay re-reads the scalar and re-stages the unsaved aggregate,
  // and nothing else changes.
  // CHECK-LABEL: Target graph pre-replay:
  // CHECK: 1/1: literal (2)
  // CHECK: 3/3: firstprivate (base-offset: 0, size: 16, staging-offset: 0, init-pending)
  // CHECK: 8/2: firstprivate (base-offset: 0, size: 16, staging-offset: 16, saved)
  // CHECK: Updates:
  // CHECK-NEXT: Arg 1 changes: data
  // CHECK-NEXT: Arg 3 changes: data
  // CHECK-NOT: Target graph pre-replay:
  //
  // The last replay: 1 * 2 + 5 + 2 + 2, then + 1 + 5 + 0 + 2 from the saved Q.
  // CHECK: Arr[1] = 19
  printf("Arr[1] = %d\n", Arr[1]);
  return 0;
}
