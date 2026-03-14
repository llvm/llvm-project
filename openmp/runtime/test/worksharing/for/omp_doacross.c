// RUN: %libomp-compile-and-run
// XFAIL: gcc-4, gcc-5, clang-3.7, clang-3.8, icc-15, icc-16
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"

#ifndef N
#define N 750
#endif

int test_doacross() {
  int i, j;
  // Allocate and zero out the matrix
  int *m = (int *)malloc(sizeof(int) * N * N);
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      m[i * N + j] = 0;
    }
  }
  // Have first row and column be 0, 1, 2, 3, etc.
  for (i = 0; i < N; ++i)
    m[i * N] = i;
  for (j = 0; j < N; ++j)
    m[j] = j;
  // Perform wavefront which results in matrix:
  // 0 1 2 3 4
  // 1 2 3 4 5
  // 2 3 4 5 6
  // 3 4 5 6 7
  // 4 5 6 7 8
  #pragma omp parallel shared(m)
  {
    int row, col;
    #pragma omp for ordered(2)
    for (row = 1; row < N; ++row) {
      for (col = 1; col < N; ++col) {
        #pragma omp ordered depend(sink : row - 1, col) depend(sink : row, col - 1)
        m[row * N + col] = m[(row - 1) * N + col] + m[row * N + (col - 1)] -
                           m[(row - 1) * N + (col - 1)];
        #pragma omp ordered depend(source)
      }
    }
  }

  // Check the bottom right element to see if iteration dependencies were held
  int retval = (m[(N - 1) * N + N - 1] == 2 * (N - 1));
  free(m);
  return retval;
}

int main(int argc, char **argv) {
  int i;
  int num_failed = 0;
  if (omp_get_max_threads() < 2)
    omp_set_num_threads(4);
  for (i = 0; i < REPETITIONS; i++) {
    if (!test_doacross()) {
      num_failed++;
    }
  }
  return num_failed;
}
