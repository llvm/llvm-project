// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// Exercises the libomp taskgraph target-data stub entry points nested in an
// 'omp taskgraph': 'target enter data', 'target update' (to/from) and
// 'target exit data' are forwarded by __kmpc_taskgraph_target_{enter,exit}_data
// and __kmpc_taskgraph_target_update to the libomptarget *_mapper calls.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 1024;
  int *a = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    a[i] = i;

#pragma omp taskgraph
  {
#pragma omp target enter data map(to : a[0 : N])

#pragma omp target
    {
      for (int i = 0; i < N; ++i)
        a[i] = a[i] + 5;
    }

    // Pull values back to the host, modify, push them back.
#pragma omp target update from(a[0 : N])
    for (int i = 0; i < N; ++i)
      a[i] *= 2;
#pragma omp target update to(a[0 : N])

#pragma omp target
    {
      for (int i = 0; i < N; ++i)
        a[i] = a[i] - 1;
    }

#pragma omp target exit data map(from : a[0 : N])
  }

  // Expected: ((i + 5) * 2) - 1 = 2 * i + 9
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != 2 * i + 9)
      ++errors;

  // CHECK: taskgraph target data errors: 0
  printf("taskgraph target data errors: %d\n", errors);
  free(a);
  return errors != 0;
}
