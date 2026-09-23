// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// Encounter order of recorded target constructs is itself a dependence when the
// construct carries no `nowait`: such a construct generates an undeferred
// (included) task, which suspends the encountering task until it completes, so
// on the recording execution the three constructs below necessarily ran one
// after another.  A replay has to preserve that.
//
// Nothing here carries a depend clause, so a replay that only honors depend
// clauses is free to run the three nodes concurrently -- and the AMDGPU plugin
// backend does exactly that, which used to read `a` back before the kernel had
// written it and miscompile every replay after the first.  The host replay path
// happened to survive it by walking a parallel region's children one at a time,
// so both replay paths are run here.

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
#pragma omp target enter data map(to : a[0 : N])

#pragma omp target
      {
        for (int i = 0; i < N; ++i)
          a[i] = a[i] + 5;
      }

#pragma omp target exit data map(from : a[0 : N])
    }
  }

  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != i + 5 * Iters)
      ++errors;

  // CHECK: taskgraph target encounter order errors: 0
  printf("taskgraph target encounter order errors: %d\n", errors);
  free(a);
  return errors != 0;
}
