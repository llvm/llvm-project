// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A `target enter data` with no matching exit inside the region maps something
// that outlives the replay.  That makes the graph's maps a change to the
// program's mapping state rather than a private arrangement, so a replay has to
// do the reference-count bookkeeping for real: the mapping has to be there when
// the region ends, and an ordinary `target exit data` outside the graph has to
// be able to remove it.
//
// It is also the case that distinguishes a replay which really performs the
// enter-data from one that merely arranged its transfers.  The `present` map on
// the kernel beside it says the mapping the graph makes is visible to the rest
// of the graph, and the value check says the `to` copy happened on this replay
// rather than once at record time -- the reference count reaching zero at the
// end of each iteration is what puts the copy back in play.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 256
#define ITERS 3

int main(void) {
  int Dev = omp_get_default_device();
  int *A = (int *)malloc(N * sizeof(int));

  int Mapped = 0, Removed = 0, Errors = 0;

  for (int It = 0; It < ITERS; ++It) {
    for (int I = 0; I < N; ++I)
      A[I] = 100 * (It + 1) + I;

#pragma omp taskgraph
    {
#pragma omp target enter data map(to : A[0 : N])
#pragma omp target map(present, alloc : A[0 : N])
      for (int I = 0; I < N; ++I)
        A[I] += 1;
    }

    if (omp_get_mapped_ptr(A, Dev))
      ++Mapped;

    // Balanced from out here, against a mapping made in there.
#pragma omp target exit data map(from : A[0 : N])

    if (!omp_get_mapped_ptr(A, Dev))
      ++Removed;

    for (int I = 0; I < N; ++I)
      if (A[I] != 100 * (It + 1) + I + 1)
        ++Errors;
  }

  // CHECK: still mapped when the region ended: 3 of 3
  printf("still mapped when the region ended: %d of %d\n", Mapped, ITERS);
  // CHECK: removed by the exit data outside: 3 of 3
  printf("removed by the exit data outside: %d of %d\n", Removed, ITERS);
  // CHECK: taskgraph unbalanced enter data errors: 0
  printf("taskgraph unbalanced enter data errors: %d\n", Errors);

  free(A);
  return Errors != 0 || Mapped != ITERS || Removed != ITERS;
}
