// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=NOGRAPH
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=GRAPH

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A region a taskgraph maps, balances and finds unmapped when it is finalized
// is one the graph keeps to itself: each replay gives it device storage of its
// own, and there is no mapping-table entry for the program to find.  So asking
// for it gets the answer an unmapped region gets -- which is also what the
// same program gets outside a graph, where the region is mapped only while its
// `target` region runs.  Only a question asked while the subgraph holding the
// region runs would come with a warning that the graph has it.  Once the
// subgraph has run the graph holds nothing, and the region is unmapped just as
// it would be outside a graph -- which is where both questions here are asked:
// B's between replays, and A's from a host task ordered after the construct.

#include <omp.h>
#include <stdio.h>

#define N 256
#define ITERS 3

static int A[N], B[N];
static int Dev, DuringHits, BetweenHits;

int main(void) {
  Dev = omp_get_default_device();
  for (int It = 0; It < ITERS; ++It) {
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : A[0 : N], B[0 : N]) depend(inout : A[0])
      for (int I = 0; I < N; ++I) {
        A[I] += 1;
        B[I] += 2;
      }
#pragma omp task depend(in : A[0])
      DuringHits += omp_get_mapped_ptr(A, Dev) != NULL;
    }
    BetweenHits += omp_target_is_present(B, Dev);
  }

  int Errors = 0;
  for (int I = 0; I < N; ++I)
    Errors += A[I] != ITERS || B[I] != 2 * ITERS;

  // NOGRAPH-NOT: asked about
  // GRAPH-NOT: asked about

  // NOGRAPH: during 0 between 0 errors 0
  // GRAPH: during 0 between 0 errors 0
  printf("during %d between %d errors %d\n", DuringHits, BetweenHits, Errors);
  return Errors != 0;
}
