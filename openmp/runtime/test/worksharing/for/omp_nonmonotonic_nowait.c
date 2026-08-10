// RUN: %libomp-compile-and-run

// The test checks nonmonotonic scheduling works correctly when threads
// may execute different loops concurrently.

#include <stdio.h>
#include <omp.h>

#define N 200
#define C 20
int main()
{
  int i, l0 = 0, l1 = 0;
  #pragma omp parallel num_threads(8)
  {
    #pragma omp for schedule(nonmonotonic:dynamic,C) nowait
    for (i = 0; i < N; ++i) {
      #pragma omp atomic
        l0++;
    }
    #pragma omp for schedule(nonmonotonic:dynamic,C) nowait
    for (i = 0; i < N * N; ++i) {
      #pragma omp atomic
        l1++;
    }
  }
  if (l0 != N || l1 != N * N) {
    printf("failed l0 = %d, l1 = %d, should be %d %d\n", l0, l1, N, N * N);
    return 1;
  } else {
    printf("passed\n");
    return 0;
  }
}
