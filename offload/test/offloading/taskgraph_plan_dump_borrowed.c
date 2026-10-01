// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_TASKGRAPH_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// The data plan of a graph whose leaves exercise the remaining argument
// properties, as dumped before each replay.
//
//   - A section through a plain pointer: the pointer itself is not mapped, so
//     its attachment has no target.
//   - `always`, which copies whoever created the mapping.
//   - `target update`, which transfers whoever mapped the range.
//   - An exit data dropping the last reference to a mapping made outside the
//     graph, which therefore performs the copy out.
//
// Range ids follow host addresses, so they are matched by size.

#include <stdio.h>
#include <stdlib.h>

int A[64], B[32];

int main(void) {
  int *P = (int *)calloc(48, sizeof(int));
  for (int It = 0; It < 3; ++It) {
#pragma omp target enter data map(alloc : B[0 : 32])
    B[0] = It;
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : P[0 : 48])
      P[0] += 1;
#pragma omp target map(always, tofrom : A[0 : 64])
      A[0] += 1;
#pragma omp target update to(B[0 : 32])
#pragma omp target map(tofrom : B[0 : 32])
      B[0] += 10;
#pragma omp target exit data map(from : B[0 : 32])
    }
  }

  // CHECK-LABEL: Target graph pre-replay:
  // CHECK: Ranges:
  // CHECK-DAG: Id: [[A:[0-9]+]], {{.*}}, Size: 256{{$}}
  // CHECK-DAG: Id: [[B:[0-9]+]], {{.*}}, Size: 128{{$}}
  // CHECK-DAG: Id: [[P:[0-9]+]], {{.*}}, Size: 192{{$}}
  // CHECK: Firstprivate staging buffer size: 0
  // CHECK: sequential {
  // CHECK-NEXT: target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 0/0: mapped (range: [[P]], offset: 0, size: 192, in (perimeter, enabled), out (perimeter, enabled))
  // CHECK-NEXT: 1/1: no-target-attach
  // CHECK-NEXT: 2/2: literal (0)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: copy-in (arg: 0)
  // CHECK-NEXT: 1: launch
  // CHECK-NEXT: 2: copy-out (arg: 0)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 3/0: mapped (range: [[A]], offset: 0, size: 256, in (perimeter, enabled), out (perimeter, enabled), always)
  // CHECK-NEXT: 4/1: literal (0)
  // CHECK-NEXT: }
  // CHECK:      update {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 5/0: mapped (range: [[B]], offset: 0, size: 128, in (internal, enabled), always)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: copy-in (arg: 0)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: target {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 6/0: mapped (range: [[B]], offset: 0, size: 128, in (perimeter, disabled), out (perimeter, disabled))
  // CHECK-NEXT: 7/1: literal (0)
  // CHECK-NEXT: }
  // CHECK:      exit data {
  // CHECK-NEXT: args {
  // CHECK-NEXT: 8/0: mapped (range: [[B]], offset: 0, size: 128, out (internal, enabled), owns-borrowed-out)
  // CHECK-NEXT: }
  // CHECK-NEXT: ops {
  // CHECK-NEXT: 0: copy-out (arg: 0)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: Updates:
  // CHECK-NEXT: Arg 0 changes: device-addr
  // CHECK-NEXT: Arg 3 changes: device-addr
  // CHECK-NEXT: Arg 5 changes: device-addr
  // CHECK-NEXT: Arg 6 changes: device-addr enable
  // CHECK-NEXT: Arg 8 changes: device-addr
  //
  // The second replay finds everything as the first left it.
  // CHECK-LABEL: Target graph pre-replay:
  // CHECK: Updates:
  // CHECK-NEXT: (no updates)
  // CHECK-NOT: Target graph pre-replay:
  //
  // CHECK: P[0] = 3, A[0] = 3, B[0] = 12
  printf("P[0] = %d, A[0] = %d, B[0] = %d\n", P[0], A[0], B[0]);
  free(P);
  return 0;
}
