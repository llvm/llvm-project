// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: omp_taskgraph_experimental

// Two mappings of one array within a single taskgraph.  Their host ranges can
// coincide, one can contain the other, or they can be disjoint -- and a graph
// has to reach the same answer as the same program without one, in every shape.
//
// What makes this worth pinning down is that a device buffer is identified by
// its host range, and the two mappings are two *different* ranges of the same
// array.  Whether they are one buffer or two is decided by the ranges alone, so
// nothing here needs to recognise that the ranges came from the same array: a
// contained range must resolve to its container at an offset (exactly what
// libomptarget's own mapping table does, reusing the enclosing entry at
// TgtPtrBegin + (HstPtrBegin - entry->HstPtrBegin)), while disjoint ranges must
// stay independent.
//
// Containment is the shape that discriminates, and the enclosing `target data`
// is what makes it observable.  The enclosing region's copy-out is the last
// thing to touch the host range, so if the inner kernel wrote a *separate*
// device buffer, that copy-out writes back the values the enclosing region
// copied in before the kernel ran and the kernel's work is silently undone.
// Laundering the result through host memory cannot rescue it: the enclosing
// mapping is live across the kernel, so there is no copy-out/copy-in pair
// between the two to carry the value.
//
// The disjoint shapes are here to keep a fix honest in the other direction: a
// coherence fix that merged ranges by proximity rather than by overlap would
// hand the second kernel a buffer that runs on into the first one's section,
// so adjacent-but-disjoint must stay two buffers.  Adjacent and gapped are
// separated because only the adjacent pair is at risk of being coalesced.
//
// Replayed several times, so a shape that is right on the recording execution
// and wrong from the first replay onwards is still caught.

#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define Q (N / 4)
#define ITERS 3

static int check(const char *Shape, int *Arr, int Lo, int Hi) {
  int Errors = 0;
  for (int I = 0; I < N; ++I) {
    int Expected = I + (I >= Lo && I < Hi ? ITERS : 0);
    if (Arr[I] != Expected)
      ++Errors;
  }
  printf("%s errors: %d\n", Shape, Errors);
  return Errors;
}

int main() {
  int *A = (int *)malloc(N * sizeof(int));
  int *B = (int *)malloc(N * sizeof(int));
  int *C = (int *)malloc(N * sizeof(int));
  int *D = (int *)malloc(N * sizeof(int));
  int *E = (int *)malloc(N * sizeof(int));
  for (int I = 0; I < N; ++I)
    A[I] = B[I] = C[I] = D[I] = E[I] = I;

  for (int It = 0; It < ITERS; ++It) {
#pragma omp taskgraph
    {
      // Coinciding: the inner range is exactly the enclosing one.
#pragma omp target data map(tofrom : A[0 : N])
      {
#pragma omp target map(tofrom : A[0 : N])
        for (int I = 0; I < N; ++I)
          A[I] += 1;
      }

      // Containing: the inner range is a strict sub-range of the enclosing
      // one, touching neither endpoint, so it cannot be mistaken for it.
#pragma omp target data map(tofrom : B[0 : N])
      {
#pragma omp target map(tofrom : B[Q : 2 * Q])
        for (int I = 0; I < 2 * Q; ++I)
          B[Q + I] += 1;
      }

      // Containing, sharing the enclosing range's start, which is the one
      // sub-range an exact-pointer buffer key does resolve to the container.
#pragma omp target data map(tofrom : C[0 : N])
      {
#pragma omp target map(tofrom : C[0 : 2 * Q])
        for (int I = 0; I < 2 * Q; ++I)
          C[I] += 1;
      }

      // Adjacent-disjoint: two ranges meeting end to start.
#pragma omp target map(tofrom : D[0 : N / 2])
      for (int I = 0; I < N / 2; ++I)
        D[I] += 1;
#pragma omp target map(tofrom : D[N / 2 : N / 2])
      for (int I = 0; I < N / 2; ++I)
        D[N / 2 + I] += 1;

      // Gapped-disjoint: a hole between the two ranges, which must stay
      // untouched.
#pragma omp target map(tofrom : E[0 : Q])
      for (int I = 0; I < Q; ++I)
        E[I] += 1;
#pragma omp target map(tofrom : E[2 * Q : Q])
      for (int I = 0; I < Q; ++I)
        E[2 * Q + I] += 1;
    }
  }

  int Errors = 0;
  // CHECK: coinciding errors: 0
  Errors += check("coinciding", A, 0, N);
  // CHECK: containing errors: 0
  Errors += check("containing", B, Q, 3 * Q);
  // CHECK: containing-at-start errors: 0
  Errors += check("containing-at-start", C, 0, 2 * Q);
  // CHECK: adjacent-disjoint errors: 0
  Errors += check("adjacent-disjoint", D, 0, N);

  int GapErrors = 0;
  for (int I = 0; I < N; ++I) {
    int Expected = I + (I < Q || (I >= 2 * Q && I < 3 * Q) ? ITERS : 0);
    if (E[I] != Expected)
      ++GapErrors;
  }
  // CHECK: gapped-disjoint errors: 0
  printf("gapped-disjoint errors: %d\n", GapErrors);
  Errors += GapErrors;

  // CHECK: taskgraph map overlap errors: 0
  printf("taskgraph map overlap errors: %d\n", Errors);
  free(E);
  free(D);
  free(C);
  free(B);
  free(A);
  return Errors != 0;
}
