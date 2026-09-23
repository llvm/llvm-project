// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: taskgraph

// A replayed 'omp taskgraph' with a 'target' that takes an *aggregate*
// firstprivate(saved:) array.  The saved value is the one captured when the
// graph was recorded; mutating the host array after each encounter must not
// affect later replays.  Recording snapshots the bytes into the captured
// kernel-arguments blob and points the argument at the snapshot, so every
// encounter privatizes from coef[j] == j + 1 however the host copy has moved
// on -- and it also keeps the argument off the recording invocation's stack.

#include <stdio.h>
#include <stdlib.h>

#define M 8

int main() {
  const int N = 256;
  const int Iters = 3;

  int coef[M];
  for (int j = 0; j < M; ++j)
    coef[j] = j + 1; // record-time value; sum = M*(M+1)/2 = 36.

  int *acc = (int *)calloc(N, sizeof(int));

  for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
    {
#pragma omp target firstprivate(saved : coef) map(tofrom : acc[0 : N])
      {
        for (int i = 0; i < N; ++i) {
          int s = 0;
          for (int j = 0; j < M; ++j)
            s += coef[j];
          acc[i] += s;
        }
      }
    }
    // Mutate the host copy after the construct; 'saved' must make replays
    // ignore this and keep using the record-time coef.
    for (int j = 0; j < M; ++j)
      coef[j] += 100;
  }

  const int Sum = M * (M + 1) / 2; // 36
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (acc[i] != Iters * Sum)
      ++errors;

  // CHECK: taskgraph saved firstprivate aggregate errors: 0
  printf("taskgraph saved firstprivate aggregate errors: %d\n", errors);
  free(acc);
  return errors != 0;
}
