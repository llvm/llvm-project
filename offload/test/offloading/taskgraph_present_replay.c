// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: omp_taskgraph_experimental

// A replayed 'omp taskgraph' whose 'target' operates on data kept live by an
// enclosing 'target data' region.  The data is present on the device for the
// whole replay, so each encounter must accumulate into the *same* live device
// allocation: no per-replay copy from the (stale) host, no premature copy-back.
// The HIP-Graph backend must bake the live device address and elide the
// implicit map copies; a wrong (graph-owned-buffer + per-replay H2D/D2H) model
// would re-seed from the never-updated host array and yield i + 1 instead of
// i + Iters.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 1024;
  const int Iters = 4;
  int *a = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    a[i] = i;

#pragma omp target data map(tofrom : a[0 : N])
  {
    for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
      {
#pragma omp target map(tofrom : a[0 : N])
        {
          for (int i = 0; i < N; ++i)
            a[i] = a[i] + 1;
        }
      }
    }
  } // end target data: copies the accumulated device values back to the host.

  // a was present throughout; each of the Iters encounters adds 1 in place, so
  // a[i] == i + Iters.
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != i + Iters)
      ++errors;

  // CHECK: taskgraph present replay errors: 0
  printf("taskgraph present replay errors: %d\n", errors);
  free(a);
  return errors != 0;
}
