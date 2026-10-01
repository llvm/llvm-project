// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env KMP_TASKGRAPH_TRACE=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefixes=CHECK,TRACE \
// RUN:   --implicit-check-not='replays on the host'

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A `target exit data` in a graph that drops a reference the graph did not
// take.  Outside a graph the decrement that removes a mapping is the one that
// copies it out, so whether this one copies depends on the mapping the replay
// finds: A's only reference is the one the graph drops, so the graph removes
// A's mapping and copies A out; B has a second reference, so B's mapping stays
// and nothing is copied until the program drops that one too.
//
// This is the shape the exit-data of a `target data` takes once the body is
// run between subgraphs, the enter-data having mapped the data in an earlier
// one -- so it is common, not contrived, and getting it wrong silently loses
// everything the body did on the device.

#include <omp.h>
#include <stdio.h>

#define N 256
#define ITERS 4

static int A[N], B[N];

int main(void) {
  int Dev = omp_get_default_device();
  int BadLast = 0, BadNotLast = 0, LeftMapped = 0, Unmapped = 0;

  for (int It = 0; It < ITERS; ++It) {
    for (int I = 0; I < N; ++I)
      A[I] = B[I] = I;

#pragma omp target enter data map(to : A[0 : N])
#pragma omp target enter data map(to : B[0 : N])
#pragma omp target enter data map(to : B[0 : N])
#pragma omp target map(present, alloc : A[0 : N], B[0 : N])
    for (int I = 0; I < N; ++I) {
      A[I] += It + 1;
      B[I] += It + 1;
    }

#pragma omp taskgraph
    {
#pragma omp target exit data map(from : A[0 : N], B[0 : N])
    }

    for (int I = 0; I < N; ++I) {
      if (A[I] != I + It + 1)
        ++BadLast;
      if (B[I] != I)
        ++BadNotLast;
    }
    LeftMapped += omp_target_is_present(A, Dev);
    Unmapped += !omp_target_is_present(B, Dev);

#pragma omp target exit data map(from : B[0 : N])
    for (int I = 0; I < N; ++I)
      if (B[I] != I + It + 1)
        ++BadNotLast;
  }

  // TRACE: Transmitted taskgraph {{.*}} subgraph

  // CHECK: last reference not copied out: 0
  printf("last reference not copied out: %d\n", BadLast);
  // CHECK: other reference copied out early: 0
  printf("other reference copied out early: %d\n", BadNotLast);
  // CHECK: left mapped: 0, unmapped early: 0
  printf("left mapped: %d, unmapped early: %d\n", LeftMapped, Unmapped);
  return BadLast + BadNotLast + LeftMapped + Unmapped != 0;
}
