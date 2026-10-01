// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_TASKGRAPH_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// The data plan of a graph whose dependences are not series-parallel, as dumped
// before each replay.
//
//   TA      TB
//   | \     |
//   |  \    |
//   TC  '-- TD
//   |
//   TE
//
// TA, TB, TC and TD form an N, the smallest shape with no series-parallel
// decomposition, so they land in an irreducible region whose edges are
// transmitted explicitly.  TE only follows TC, so the two of them are carved
// into a sequential region, which is one of the irreducible region's children.
// Each child lists the children it is a predecessor of, a leaf and a region
// alike.
//
// `nowait` keeps the depend clauses the only edges: an undeferred target is
// ordered after everything encountered before it, which would make the graph a
// total order.
//
// Each leaf maps an array of its own size, so its range is identified by size.

#include <stdio.h>

int A[1], B[2], C[3], D[4], E[5];

int main(void) {
  int Da, Db, Dc;
  for (int It = 0; It < 2; ++It) {
#pragma omp taskgraph
    {
#pragma omp target nowait map(from : A[0 : 1]) depend(out : Da)
      A[0] = 1;
#pragma omp target nowait map(from : B[0 : 2]) depend(out : Db)
      B[0] = 2;
#pragma omp target nowait map(from : C[0 : 3]) depend(in : Da) depend(out : Dc)
      C[0] = 3;
#pragma omp target nowait map(from : D[0 : 4]) depend(in : Da, Db)
      D[0] = 4;
#pragma omp target nowait map(from : E[0 : 5]) depend(in : Dc)
      E[0] = 5;
    }
  }

  // The children are numbered in the order libomp carves them, which puts the
  // sequential region first.
  //
  // CHECK-LABEL: Target graph pre-replay:
  // CHECK: Ranges:
  // CHECK-DAG: Id: [[A:[0-9]+]], {{.*}}, Size: 4{{$}}
  // CHECK-DAG: Id: [[B:[0-9]+]], {{.*}}, Size: 8{{$}}
  // CHECK-DAG: Id: [[C:[0-9]+]], {{.*}}, Size: 12{{$}}
  // CHECK-DAG: Id: [[D:[0-9]+]], {{.*}}, Size: 16{{$}}
  // CHECK-DAG: Id: [[E:[0-9]+]], {{.*}}, Size: 20{{$}}
  // CHECK: Firstprivate staging buffer size: 0
  // CHECK: attach staging buffer size: 0
  // CHECK: irreducible {
  // CHECK-NEXT: [0] sequential {
  // CHECK-NEXT: target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 0/0: mapped (range: [[C]], offset: 0, size: 12, out (perimeter, enabled))
  // CHECK-NEXT: 1/1: literal (0)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: launch
  // CHECK-NEXT: 1: copy-out (arg: 0)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 2/0: mapped (range: [[E]], offset: 0, size: 20, out (perimeter, enabled))
  // CHECK-NEXT: 3/1: literal (0)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: launch
  // CHECK-NEXT: 1: copy-out (arg: 0)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: successors {
  // CHECK-EMPTY:
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: [1] target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 4/0: mapped (range: [[A]], offset: 0, size: 4, out (perimeter, enabled))
  // CHECK-NEXT: 5/1: literal (0)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: launch
  // CHECK-NEXT: 1: copy-out (arg: 0)
  // CHECK-NEXT: }
  // CHECK-NEXT: successors {
  // CHECK-NEXT: -> [0], [2]{{$}}
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: [2] target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 6/0: mapped (range: [[D]], offset: 0, size: 16, out (perimeter, enabled))
  // CHECK-NEXT: 7/1: literal (0)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: launch
  // CHECK-NEXT: 1: copy-out (arg: 0)
  // CHECK-NEXT: }
  // CHECK-NEXT: successors {
  // CHECK-EMPTY:
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: [3] target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 8/0: mapped (range: [[B]], offset: 0, size: 8, out (perimeter, enabled))
  // CHECK-NEXT: 9/1: literal (0)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: launch
  // CHECK-NEXT: 1: copy-out (arg: 0)
  // CHECK-NEXT: }
  // CHECK-NEXT: successors {
  // CHECK-NEXT: -> [2]{{$}}
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: Updates:
  // CHECK-NEXT: Arg 0 changes: device-addr
  // CHECK-NEXT: Arg 2 changes: device-addr
  // CHECK-NEXT: Arg 4 changes: device-addr
  // CHECK-NEXT: Arg 6 changes: device-addr
  // CHECK-NEXT: Arg 8 changes: device-addr
  // CHECK-NOT: Target graph pre-replay:
  //
  // CHECK: A[0] + B[0] + C[0] + D[0] + E[0] = 15
  printf("A[0] + B[0] + C[0] + D[0] + E[0] = %d\n",
         A[0] + B[0] + C[0] + D[0] + E[0]);
  return 0;
}
