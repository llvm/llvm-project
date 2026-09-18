// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=NOGRAPH
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=GRAPH

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A pointee a taskgraph would keep to itself, attached to a pointer the
// program has mapped.  The device copy of G.P outlives every replay, so were
// the pointee in storage of the graph's own it would be left pointing at
// memory freed when the replay ended; the graph maps that pointee through the
// mapping table instead, as it would a region found mapped.  L.P's pointee has
// no such pointer -- L is mapped by the graph alone -- so it stays the graph's.
//
// Either way the graph holds neither once the subgraph has run, so the lookup
// APIs, asked from a host task ordered after the construct, find L.P's pointee
// unmapped as they would outside a graph, and nothing warns that the graph has
// it.  Only the answer about L.P's pointee is compared: G.P's is mapped by the
// `target enter data` of G for as long as G is.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 256
#define ITERS 3

struct S {
  int *P;
  int Tag;
};

static struct S G;
static int *DataG, *DataL;
static int Dev, Hits;

static void step(void) {
  struct S L;
  L.P = DataL;
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : G.P[0 : N]) map(tofrom : L.P[0 : N])           \
    depend(out : G)
    for (int I = 0; I < N; ++I) {
      G.P[I] += 1;
      L.P[I] += 2;
    }
#pragma omp task depend(in : G)
    {
      (void)omp_get_mapped_ptr(DataG, Dev);
      Hits += omp_target_is_present(DataL, Dev);
    }
  }
}

int main(void) {
  DataG = (int *)calloc(N, sizeof(int));
  DataL = (int *)calloc(N, sizeof(int));
  G.P = DataG;
  Dev = omp_get_default_device();

#pragma omp target enter data map(to : G)
  for (int It = 0; It < ITERS; ++It)
    step();
#pragma omp target exit data map(release : G)

  int Errors = 0;
  for (int I = 0; I < N; ++I)
    Errors += DataG[I] != ITERS || DataL[I] != 2 * ITERS;

  // NOGRAPH-NOT: asked about
  // GRAPH-NOT: asked about

  // NOGRAPH: hits 0 errors 0
  // GRAPH: hits 0 errors 0
  printf("hits %d errors %d\n", Hits, Errors);
  free(DataL);
  free(DataG);
  return Errors != 0;
}
