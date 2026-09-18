// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: omp_taskgraph_experimental

// A graph a device backend is expected to decline, replayed with the backend
// enabled.  A user-defined mapper expands to an entry list the *runtime*
// computes: the generated mapper emits an optional leading entry, then one
// component list per element of the section it was invoked on, so how many map
// entries there are is not a property of the recording.  A graph's topology is
// fixed when it is recorded, so the plan declines and libomp replays the graph
// on the host, issuing each recorded target region as its dataflow walk reaches
// it -- and the mapper then runs exactly as it does outside a graph.
//
// Declining is a supported outcome, not a failure: the point of this test is
// that the answer is still right, so a backend that grows support for these maps
// and stops declining keeps passing.

#include <stdio.h>
#include <stdlib.h>

struct Vec {
  int *P;
  int Len;
};

#pragma omp declare mapper(struct Vec v) map(v, v.P[0 : v.Len])

int main() {
  const int N = 1024;
  const int Iters = 3;
  struct Vec V;
  V.P = (int *)malloc(N * sizeof(int));
  V.Len = N;
  for (int i = 0; i < N; ++i)
    V.P[i] = i;

  for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : V)
      {
        for (int i = 0; i < V.Len; ++i)
          V.P[i] = V.P[i] + 1;
      }
    }
  }

  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (V.P[i] != i + Iters)
      ++errors;

  // CHECK: taskgraph decline errors: 0
  printf("taskgraph decline errors: %d\n", errors);
  free(V.P);
  return errors != 0;
}
