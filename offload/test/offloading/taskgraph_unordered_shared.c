// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_INFO=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --implicit-check-not='replays on the host'

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// Unordered leaves sharing data that the graph keeps on the device, unlike
// taskgraph_unordered_copy_decline.c.
//
// - Two tasks reading a shared input, each mapping it `to` for itself, inside
//   a `target data` that already maps it.  Each copies the input in before its
//   own kernel, and the other's copy writes the same bytes.
// - Two kernels reaching data a `target data` has mapped through a pointer,
//   writing disjoint halves.  Nothing in the graph copies that data.

#include <stdio.h>
#include <stdlib.h>

#define N 4096
#define ITERS 4

int main(void) {
  int W[N], Out1[N], Out2[N];
  int *T = (int *)malloc(N * sizeof(int));
  int Dep1, Dep2, Errors = 0;

  for (int I = 0; I < N; ++I)
    W[I] = I;
#pragma omp target data map(to : W[0 : N])
  for (int It = 0; It < ITERS; ++It) {
#pragma omp taskgraph
    {
#pragma omp target map(to : W[0 : N]) map(from : Out1[0 : N])                  \
    nowait depend(out : Dep1)
      for (int I = 0; I < N; ++I)
        Out1[I] = W[I] + It;
#pragma omp target map(to : W[0 : N]) map(from : Out2[0 : N])                  \
    nowait depend(out : Dep2)
      for (int I = 0; I < N; ++I)
        Out2[I] = 2 * W[I] + It;
    }
#pragma omp taskwait
    for (int I = 0; I < N; ++I)
      Errors += Out1[I] != I + It || Out2[I] != 2 * I + It;
  }

  for (int I = 0; I < N; ++I)
    T[I] = 0;
#pragma omp target data map(tofrom : T[0 : N])
  {
    for (int It = 0; It < ITERS; ++It) {
#pragma omp taskgraph
      {
#pragma omp target nowait depend(out : Dep1)
        for (int I = 0; I < N / 2; ++I)
          T[I] += 1;
#pragma omp target nowait depend(out : Dep2)
        for (int I = N / 2; I < N; ++I)
          T[I] += 2;
      }
#pragma omp taskwait
    }
  }
  for (int I = 0; I < N; ++I)
    Errors += T[I] != (I < N / 2 ? ITERS : 2 * ITERS);

  // CHECK: taskgraph unordered shared errors: 0
  printf("taskgraph unordered shared errors: %d\n", Errors);
  free(T);
  return Errors != 0;
}
