// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env KMP_TASKGRAPH_TRACE=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic \
// RUN:   --check-prefixes=CHECK,GRAPH \
// RUN:   --implicit-check-not='replays on the host'

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A graph holding the body of a 'target data' is replayed with the target
// constructs either side of the body on the device, as subgraphs, and the body
// run by libomp between them.  The body is code the graph cannot see, and it
// finds the construct's data the way any code does: through the mapping table.
// So what the subgraphs before it map, and leave for those after it to unmap,
// is published there rather than kept as a subgraph's private storage.
//
// Nothing outside the graph maps A, B or C, so every mapping the body finds is
// one the graph made:
//  - A is the construct's own data: the use_device_ptr conversion,
//    omp_get_mapped_ptr and a nested 'present' region must all find it, and
//    agree on where it is.
//  - B is mapped by a 'target enter data' before the construct and unmapped
//    by a 'target exit data' after it, so it is present in the body too.
//  - C is mapped only by a 'target' region that has finished before the body
//    starts, so it is not present there, and stays the graph's own.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 64
#define ITERS 4

int main() {
  int Dev = omp_get_default_device();
  int *A = (int *)malloc(N * sizeof(int));
  int *B = (int *)malloc(N * sizeof(int));
  int *C = (int *)malloc(N * sizeof(int));
  int BadUdp = 0, BadA = 0, BadB = 0, BadC = 0, BadData = 0;

  // Within the body A names the device copy, so the host array is looked up
  // through another name.
  int *HostA = A;

  for (int It = 0; It < ITERS; ++It) {
    for (int I = 0; I < N; ++I)
      A[I] = B[I] = C[I] = 0;

#pragma omp taskgraph
    {
#pragma omp target map(tofrom : C[0 : N])
      for (int I = 0; I < N; ++I)
        C[I] = 1;
#pragma omp target enter data map(to : B[0 : N])
#pragma omp target data map(tofrom : A[0 : N]) use_device_ptr(A)
      {
        void *MappedA = omp_get_mapped_ptr(HostA, Dev);
        if (!MappedA || A != MappedA)
          ++BadUdp;
#pragma omp target is_device_ptr(A)
        for (int I = 0; I < N; ++I)
          A[I] = It + 1;
        if (!omp_target_is_present(B, Dev) || !omp_get_mapped_ptr(B, Dev))
          ++BadB;
        if (omp_target_is_present(C, Dev))
          ++BadC;
#pragma omp target map(present, tofrom : B[0 : N])
        for (int I = 0; I < N; ++I)
          B[I] = It + 2;
      }
#pragma omp target exit data map(from : B[0 : N])
    }

    for (int I = 0; I < N; ++I)
      if (A[I] != It + 1 || B[I] != It + 2 || C[I] != 1)
        ++BadData;
    if (omp_target_is_present(A, Dev) || omp_target_is_present(B, Dev))
      ++BadA;
  }

  // GRAPH: Transmitted taskgraph
  // CHECK: use_device_ptr mismatches: 0
  printf("use_device_ptr mismatches: %d\n", BadUdp);
  // CHECK: B absent in body: 0
  printf("B absent in body: %d\n", BadB);
  // CHECK: C present in body: 0
  printf("C present in body: %d\n", BadC);
  // CHECK: left mapped after replay: 0
  printf("left mapped after replay: %d\n", BadA);
  // CHECK: data errors: 0
  printf("data errors: %d\n", BadData);
  free(A);
  free(B);
  free(C);
  return (BadUdp + BadA + BadB + BadC + BadData) != 0;
}
