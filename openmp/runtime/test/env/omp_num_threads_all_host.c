// RUN: %libomp-compile && env OMP_NUM_THREADS_ALL=8 %libomp-run
//
// OpenMP 6.0: host falls through to `_ALL` when `<ENV>` is unset.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int max = omp_get_max_threads();
  if (max != 8) {
    fprintf(stderr, "FAIL: omp_get_max_threads()=%d, expected 8\n", max);
    return 1;
  }
  return 0;
}
