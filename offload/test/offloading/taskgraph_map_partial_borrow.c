// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-fail-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefix=NOGRAPH
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-fail-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefix=GRAPH

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// Fails until a plugin claims taskgraphs, which the AMDGPU backend is the
// first to do.
// XFAIL: *

// A mapped range inside a taskgraph that only *partially* overlaps one mapped
// outside it.  Asking whether a range is mapped is a question about whole-range
// containment, so a partial overlap answers no, and a graph that reads that as
// "not mapped" takes the range for its own and gives it a buffer -- one which
// aliases the outer mapping's device memory over the part they share, with
// nothing keeping the two coherent.  Nor is there an address to borrow instead,
// since the mapping does not cover the whole range.
//
// So there is no correct silent answer, and libomptarget does not attempt one:
// outside a graph this shape is refused as an explicit extension of mapped
// data.  The graph declines the replay rather than guessing, which hands it to
// the host path and so to that same refusal.  Both runs below therefore expect
// to die, of the same error, and the point of running both is that the graph
// refuses the same program the plain path refuses.
//
// The range is unmapped when the graph is finalized and balanced within it, so
// the graph keeps it on the device for itself rather than in the mapping table.
// The outer mapping is made between replays, when the graph holds nothing, and
// the replay then finds its range partly mapped and declines.
//
// The overlap is arranged to arrive between replays rather than at record time.
// A partial overlap present while the graph is recorded is already caught, by
// libomptarget itself, on the recording execution -- that pass maps normally --
// so the graph never reaches the point of resolving it.  It is the mapping that
// appears afterwards that a graph resolving its ranges once would miss.

#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define OUTER (N / 2)    // outer maps [0, 512)
#define INNER_LO (N / 4) // graph maps [256, 768): overlaps the outer tail
#define ITERS 3

int main() {
  int *A = (int *)malloc(N * sizeof(int));
  for (int I = 0; I < N; ++I)
    A[I] = I;

  for (int It = 0; It < ITERS; ++It) {
    // Absent while the graph is recorded, so the graph takes its range for its
    // own, and present for the replays.
#pragma omp target data map(tofrom : A[0 : OUTER]) if (It > 0)
    {
#pragma omp taskgraph
      {
#pragma omp target map(tofrom : A[INNER_LO : OUTER])
        for (int I = 0; I < OUTER; ++I)
          A[INNER_LO + I] += 1;
      }
    }
  }

  // NOGRAPH: explicit extension not allowed
  // NOGRAPH: fatal error
  // GRAPH-NOT: overlaps memory
  // GRAPH: taskgraph keeps [{{.*}}) on the device for itself, and it is now partly mapped outside the graph
  // GRAPH: taskgraph {{[0-9]+}} replays on the host this time
  // GRAPH: explicit extension not allowed
  // GRAPH: fatal error
  printf("reached the end\n");
  // NOGRAPH-NOT: reached the end
  // GRAPH-NOT: reached the end
  free(A);
  return 0;
}
