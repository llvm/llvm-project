// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_TASKGRAPH_REPORT_DRIFT=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=DRIFT
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=QUIET

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// Fails until a plugin claims taskgraphs, which the AMDGPU backend is the
// first to do.
// XFAIL: *

// A graph is recorded once and replayed many times, but the mapping state it
// was recorded against belongs to the program, not to the graph, and can differ
// on every replay.  A region the graph found mapped may be gone, and one it
// found unmapped may have acquired a mapping, so whether the graph should use
// somebody else's device buffer or one of its own is a question that has to be
// re-asked per replay rather than settled once when the graph is built.
//
// An `if` clause on an enclosing `target data` is what varies it without
// varying the graph: the construct, and so the graph, is the same one every
// iteration, while the mapping around it comes and goes.
//
// Both directions of the change are wrong in the same way -- two device copies
// of one host range -- but they fail through different paths.  Where the graph
// believes a region is unmapped but it is now mapped, the graph copies through a
// buffer of its own while the enclosing region holds a second copy that never
// sees the kernel, and the enclosing region's copy-out then writes that older
// copy back over the host.  Where the graph believes a region is mapped but the
// mapping has since gone, the graph is left addressing a device allocation that
// was freed underneath it.
//
// In the first direction, a region unmapped when the graph is finalized, and
// balanced within it, is one the graph keeps to itself, in storage of its own
// with no mapping-table entry.  Finding it mapped from outside on a later
// replay, by a mapping that covers it, the graph borrows that mapping for the
// replay as it would a region it found mapped: the device address is the
// mapping's, and the graph's own copies are off, the mapping's standing in for
// them.  Nothing is declined or warned about.

#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define ITERS 3

int main() {
  int *A = (int *)malloc(N * sizeof(int));
  int *B = (int *)malloc(N * sizeof(int));
  for (int I = 0; I < N; ++I)
    A[I] = B[I] = I;

  // Unmapped when the graph is built, mapped on every replay.
  for (int It = 0; It < ITERS; ++It) {
#pragma omp target data map(tofrom : A[0 : N]) if (It > 0)
    {
#pragma omp taskgraph
      {
#pragma omp target map(tofrom : A[0 : N])
        for (int I = 0; I < N; ++I)
          A[I] += 1;
      }
    }
  }

  int AErrors = 0;
  for (int I = 0; I < N; ++I)
    if (A[I] != I + ITERS)
      ++AErrors;
  // CHECK: unmapped when recorded errors: 0
  printf("unmapped when recorded errors: %d\n", AErrors);

  // Mapped when the graph is built, unmapped on every replay.
  for (int It = 0; It < ITERS; ++It) {
#pragma omp target data map(tofrom : B[0 : N]) if (It == 0)
    {
#pragma omp taskgraph
      {
#pragma omp target map(tofrom : B[0 : N])
        for (int I = 0; I < N; ++I)
          B[I] += 1;
      }
    }
  }

  int BErrors = 0;
  for (int I = 0; I < N; ++I)
    if (B[I] != I + ITERS)
      ++BErrors;
  // CHECK: mapped when recorded errors: 0
  printf("mapped when recorded errors: %d\n", BErrors);

  // CHECK: taskgraph map drift errors: 0
  printf("taskgraph map drift errors: %d\n", AErrors + BErrors);

  // Right answers alone would not show the drift was taken, so the count is
  // pinned too.  Three, not four: the first loop's two replays each follow a
  // mapping appearing, but the second loop's mapping goes away once and then
  // stays away, so its later replay finds the state it left behind and the
  // check costs nothing.  A fourth would mean that short-circuit is broken.
  // DRIFT-COUNT-3: mapping drift, re-resolving ranges
  // DRIFT-NOT: mapping drift
  //
  // Every replay runs on the device.
  // QUIET-NOT: omptarget message
  free(B);
  free(A);
  return (AErrors + BErrors) != 0;
}
