// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: omp_taskgraph_experimental

// A 'target' region lexically nested in an 'omp taskgraph', encountered once,
// so the graph is only recorded: __kmpc_taskgraph_target records the region
// and launches it through __tgt_target_kernel, producing the same result as an
// ordinary launch.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 1024;
  int *a = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    a[i] = i;

#pragma omp taskgraph
  {
#pragma omp target map(tofrom : a[0 : N])
    {
      for (int i = 0; i < N; ++i)
        a[i] = a[i] * 2 + 1;
    }
  }

  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != 2 * i + 1)
      ++errors;

  // CHECK: taskgraph target errors: 0
  printf("taskgraph target errors: %d\n", errors);
  free(a);
  return errors != 0;
}
