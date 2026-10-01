// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_INFO=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A graph mapping something the data plan cannot express is replayed by libomp
// on the host instead, which is correct but gives up the device graph.  Each
// such map must say so under LIBOMPTARGET_INFO, naming what was wrong with it,
// rather than falling back without a trace; and the results must still be
// right.

#include <stdio.h>
#include <stdlib.h>

#define N 64

typedef struct {
  int Len;
  int *Data;
} VecTy;

#pragma omp declare mapper(VecTy V) map(V, V.Data[0 : V.Len])

int main() {
  const int Iters = 3;
  int errors = 0;

  VecTy V;
  V.Len = N;
  V.Data = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    V.Data[i] = i;

  // CHECK: taskgraph {{[0-9]+}} replays on the host: argument {{[0-9]+}} of leaf 0 ({{.*}}) uses a user-defined mapper
  for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : V)
      for (int i = 0; i < V.Len; ++i)
        V.Data[i] += 1;
    }
  }
  for (int i = 0; i < N; ++i)
    if (V.Data[i] != i + Iters)
      ++errors;

  // A strided update copies every other element.  The odd ones keep whatever
  // the device last had, which is what the region wrote before the loop.
  int A[N];
  for (int i = 0; i < N; ++i)
    A[i] = 0;
#pragma omp target data map(tofrom : A[0 : N])
  {
#pragma omp target
    for (int i = 0; i < N; ++i)
      A[i] = 100;
    for (int it = 0; it < Iters; ++it) {
      for (int i = 0; i < N; ++i)
        A[i] = it;
      // CHECK: taskgraph {{[0-9]+}} replays on the host: argument {{[0-9]+}} of leaf 0 ({{.*}}) is a non-contiguous transfer
#pragma omp taskgraph
      {
#pragma omp target update to(A[0 : N / 2 : 2])
      }
    }
  }
  for (int i = 0; i < N; ++i)
    if (A[i] != (i % 2 ? 100 : Iters - 1))
      ++errors;

  // CHECK: taskgraph decline info errors: 0
  printf("taskgraph decline info errors: %d\n", errors);
  free(V.Data);
  return errors != 0;
}
