// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// The 'omp target data' *block* construct nested in an 'omp taskgraph'.  It is
// recorded as two nodes, an enter-data action and an exit-data action, with the
// body between them; the encounter-order edges the runtime builds for
// undeferred nodes are what keeps the pair around the body and one region's
// exit ahead of the next region's enter.  Before it was recorded at all, the
// mapping was established by inline host mapper calls that a replay never
// re-runs, so the maps were silently lost from the second encounter onwards.
//
// 'present' on the inner target is what makes that loss a failure rather than a
// coincidence: without it the inner target's own map would allocate a fresh
// device buffer per replay and still produce the right answer, hiding the
// missing region.  With it, a replay whose enter-data action did not run (or
// ran after the body) fails the present check outright.  Because the region is
// present, the inner target's map performs no transfer of its own, so the
// copy-back also comes from the exit-data action alone.
//
// Two sibling regions in the one graph, with non-commuting bodies, so that a
// replay which lets the first region's exit slip past the second region's enter
// gets the arithmetic wrong rather than merely running out of order.
//
// The last RUN also asks the AMDGPU plugin to claim the graph, so its lowering
// of the recorded data nodes (whether it treats the mapping as a graph-owned
// buffer or declines the graph and lets libomp replay it) has to reach the same
// answer.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 1024;
  const int Iters = 3;
  int *a = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    a[i] = i;

  for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
    {
#pragma omp target data map(tofrom : a[0 : N])
      {
#pragma omp target map(present, tofrom : a[0 : N])
        {
          for (int i = 0; i < N; ++i)
            a[i] = a[i] + 1;
        }
      }

#pragma omp target data map(tofrom : a[0 : N])
      {
#pragma omp target map(present, tofrom : a[0 : N])
        {
          for (int i = 0; i < N; ++i)
            a[i] = a[i] * 2;
        }
      }
    }
  }

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    int expected = i;
    for (int it = 0; it < Iters; ++it)
      expected = (expected + 1) * 2;
    if (a[i] != expected)
      ++errors;
  }

  // CHECK: taskgraph target data block errors: 0
  printf("taskgraph target data block errors: %d\n", errors);
  free(a);
  return errors != 0;
}
