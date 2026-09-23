// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env KMP_TASKGRAPH_TRACE=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// A recorded target construct without `nowait` generates an undeferred
// (included) task, so encounter order is a dependence and the recorded schedule
// has to be a chain.  With `nowait` the task is deferrable, program order says
// nothing, and independent targets stay in one parallel region -- which is the
// only way to get a parallel recorded graph out of a sequence of targets, so it
// is worth pinning down that the encounter-order chaining does not swallow it.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 256;
  int *a = (int *)malloc(N * sizeof(int));
  int *b = (int *)malloc(N * sizeof(int));
  int *c = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    a[i] = b[i] = c[i] = i;

  for (int it = 0; it < 2; ++it) {
    // No nowait: three undeferred tasks, so a chain.
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : a[0 : N])
      for (int i = 0; i < N; ++i)
        a[i] += 1;
#pragma omp target map(tofrom : b[0 : N])
      for (int i = 0; i < N; ++i)
        b[i] += 1;
#pragma omp target map(tofrom : c[0 : N])
      for (int i = 0; i < N; ++i)
        c[i] += 1;
    }

    // nowait, disjoint data, no depend clauses: nothing orders these.
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : a[0 : N]) nowait
      for (int i = 0; i < N; ++i)
        a[i] += 1;
#pragma omp target map(tofrom : b[0 : N]) nowait
      for (int i = 0; i < N; ++i)
        b[i] += 1;
#pragma omp target map(tofrom : c[0 : N]) nowait
      for (int i = 0; i < N; ++i)
        c[i] += 1;
    }
  }

  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] != i + 4 || b[i] != i + 4 || c[i] != i + 4)
      ++errors;

  // CHECK: Processed taskgraph
  // CHECK-NEXT: sequential {
  // CHECK-NEXT:   target: 0x{{[[:xdigit:]]+}}
  // CHECK-NEXT:   target: 0x{{[[:xdigit:]]+}}
  // CHECK-NEXT:   target: 0x{{[[:xdigit:]]+}}
  // CHECK-NEXT: }

  // CHECK: Processed taskgraph
  // CHECK-NEXT: parallel {
  // CHECK-NEXT:   target: 0x{{[[:xdigit:]]+}}
  // CHECK-NEXT:   target: 0x{{[[:xdigit:]]+}}
  // CHECK-NEXT:   target: 0x{{[[:xdigit:]]+}}
  // CHECK-NEXT: }

  // CHECK: taskgraph target nowait schedule errors: 0
  printf("taskgraph target nowait schedule errors: %d\n", errors);
  free(c);
  free(b);
  free(a);
  return errors != 0;
}
