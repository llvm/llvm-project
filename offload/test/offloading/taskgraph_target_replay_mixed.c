// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// A mixed host + target 'omp taskgraph', re-encountered several times.  A host
// 'task' (depend out) runs before a 'target' kernel (depend in) each encounter.
// On replay the graph is owned by libomptarget: the host subtree is run by
// calling back into libomp (the host_cb / __kmp_taskgraph_host_exec), while the
// target kernel is re-issued through the standard libomptarget launch path.
// The dependency ordering (host then device) must be preserved every time.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 1024;
  const int Iters = 4;
  int *a = (int *)malloc(N * sizeof(int));
  int *ref = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    a[i] = ref[i] = i;

  for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
    {
#pragma omp task depend(out : a[0 : N]) shared(a)
      {
        for (int i = 0; i < N; ++i)
          a[i] = a[i] + 1; // host side, runs first
      }
#pragma omp target map(tofrom : a[0 : N]) depend(in : a[0 : N])
      {
        for (int i = 0; i < N; ++i)
          a[i] = a[i] * 2; // device side, runs second
      }
    }
  }

  // Host reference: same (x+1)*2 transform applied Iters times.
  for (int it = 0; it < Iters; ++it)
    for (int i = 0; i < N; ++i)
      ref[i] = (ref[i] + 1) * 2;

  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != ref[i])
      ++errors;

  // CHECK: taskgraph mixed replay errors: 0
  printf("taskgraph mixed replay errors: %d\n", errors);
  free(a);
  free(ref);
  return errors != 0;
}
