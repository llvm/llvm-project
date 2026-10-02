// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env KMP_TASKGRAPH_TRACE=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic \
// RUN:   --check-prefixes=CHECK,GRAPH \
// RUN:   --implicit-check-not='replays on the host'

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// Fails until a plugin claims taskgraphs, which the AMDGPU backend is the
// first to do.
// XFAIL: *

// The body of a recorded 'target data' is a task, which converts the
// use_device_ptr list item itself whenever it starts.  So each replay sees the
// device address the list item has now, not the one it had when the graph was
// recorded: the enclosing mapping is made afresh around every replay, behind a
// growing allocation, so that it is unlikely to land in the same place twice.
//
// Within the body the list item is an ordinary variable holding the device
// address.  A nested task finds it there under the default data-sharing rules
// (firstprivate) and when it is shared, and a nested target can hand it to
// the kernel with is_device_ptr.  The nested constructs are part of the body
// task's region, not the taskgraph's, so they are not recorded but run afresh
// on each replay.
//
// The construct's enter-data and exit-data still go to the device plugin, as
// subgraphs either side of the body, which libomp runs.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 64
#define ITERS 4

int main() {
  int Dev = omp_get_default_device();
  int *A = (int *)malloc(N * sizeof(int));
  int *Pad[ITERS];
  int BadFp = 0, BadSh = 0, BadTgt = 0;

  for (int It = 0; It < ITERS; ++It) {
    for (int I = 0; I < N; ++I)
      A[I] = -1;
    // Something new on the device ahead of A each time, so that A's device
    // copy moves.
    Pad[It] = (int *)malloc((It + 1) * 4096);
#pragma omp target enter data map(alloc : Pad[It][0 : (It + 1) * 1024])
#pragma omp target enter data map(to : A[0 : N])
    void *WantA = omp_get_mapped_ptr(A, Dev);
    void *SeenFp = NULL, *SeenSh = NULL;

#pragma omp taskgraph
    {
#pragma omp target data map(tofrom : A[0 : N]) use_device_ptr(A)
      {
#pragma omp task shared(SeenFp)
        SeenFp = A;
#pragma omp task shared(A, SeenSh)
        SeenSh = A;
#pragma omp target is_device_ptr(A)
        for (int I = 0; I < N; ++I)
          A[I] = It;
      }
    }

#pragma omp target exit data map(from : A[0 : N])
    if (SeenFp != WantA)
      ++BadFp;
    if (SeenSh != WantA)
      ++BadSh;
    if (A[0] != It || A[N - 1] != It)
      ++BadTgt;
  }
  for (int It = 0; It < ITERS; ++It) {
#pragma omp target exit data map(release : Pad[It][0 : (It + 1) * 1024])
    free(Pad[It]);
  }

  // GRAPH: Transmitted taskgraph
  // CHECK: firstprivate task mismatches: 0
  printf("firstprivate task mismatches: %d\n", BadFp);
  // CHECK: shared task mismatches: 0
  printf("shared task mismatches: %d\n", BadSh);
  // CHECK: is_device_ptr target mismatches: 0
  printf("is_device_ptr target mismatches: %d\n", BadTgt);
  free(A);
  return (BadFp + BadSh + BadTgt) != 0;
}
