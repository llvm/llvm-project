// RUN: %libomptarget-compileopt-and-run-generic

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1024 * 1024 * 256)

int main(int argc, char *argv[]) {
  int *data = (int *)malloc(N * sizeof(int));
  double duration = 0.0;

#pragma omp target map(from : data[0 : N]) map(from : duration)
  {
    double start = omp_get_wtime();
    for (int i = 0; i < N; ++i)
      data[i] = i;
    double end = omp_get_wtime();
    duration = end - start;
  }
  assert(duration > 0.0);
  free(data);
  return 0;
}
