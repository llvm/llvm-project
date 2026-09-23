// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// Exercises the 'replayable' clause on target / target-data constructs that are
// NOT lexically nested inside an 'omp taskgraph'. The compiler routes these to
// the __kmpc_taskgraph_target* entry points; since no taskgraph is being
// recorded here, the runtime stubs must degrade to the ordinary libomptarget
// path (honoring depend/nowait) and produce the same result as a plain target.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 1024;
  int *a = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    a[i] = i;

  // replayable with no argument (constant-true): single taskgraph entry-point
  // call that degrades to a normal launch since nothing is recording.
#pragma omp target map(tofrom : a[0 : N]) replayable
  {
    for (int i = 0; i < N; ++i)
      a[i] = a[i] * 2 + 1;
  }

  // replayable(expr) with a true condition: takes the taskgraph entry-point
  // branch, which degrades to a normal launch.
  int t = 1;
#pragma omp target map(tofrom : a[0 : N]) replayable(t)
  {
    for (int i = 0; i < N; ++i)
      a[i] = a[i] + 3;
  }

  // replayable(expr) with a false condition: takes the ordinary launch branch.
  int f = 0;
#pragma omp target map(tofrom : a[0 : N]) replayable(f)
  {
    for (int i = 0; i < N; ++i)
      a[i] = a[i] - 2;
  }

  // replayable target with depend + nowait outside a taskgraph: a replayable
  // target executes synchronously, so no hidden helper task is created. The
  // dependences are forwarded to __kmpc_taskgraph_target (which satisfies them)
  // and the nowait is dropped, so the following taskwait is a no-op.
#pragma omp target map(tofrom : a[0 : N]) depend(inout : a[0]) nowait replayable
  {
    for (int i = 0; i < N; ++i)
      a[i] = a[i] + 10;
  }
#pragma omp taskwait

  // replayable target-data (enter / update / exit) outside a taskgraph.
#pragma omp target enter data map(to : a[0 : N]) replayable
#pragma omp target map(alloc : a[0 : N])
  {
    for (int i = 0; i < N; ++i)
      a[i] = a[i] * 3;
  }
#pragma omp target update from(a[0 : N]) replayable(t)
  for (int i = 0; i < N; ++i)
    a[i] += 1;
#pragma omp target update to(a[0 : N]) replayable
#pragma omp target map(alloc : a[0 : N])
  {
    for (int i = 0; i < N; ++i)
      a[i] = a[i] - 4;
  }
#pragma omp target exit data map(from : a[0 : N]) replayable

  // Trace per element:
  //   i
  //   *2+1   -> 2i+1
  //   +3     -> 2i+4
  //   -2     -> 2i+2
  //   +10    -> 2i+12
  //   *3     -> 6i+36
  //   +1     -> 6i+37
  //   -4     -> 6i+33
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != 6 * i + 33)
      ++errors;

  // CHECK: replayable target errors: 0
  printf("replayable target errors: %d\n", errors);
  free(a);
  return errors != 0;
}
