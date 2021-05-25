// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <omp.h>
#include "omp_testsuite.h"

int test_omp_critical(int iter) {
  int sum;
  int known_sum;

  sum = 0;
#pragma omp parallel
  {
    int mysum = 0;
    int i;
#pragma omp for
    for (i = 0; i < 1000; i++)
      mysum = mysum + i;

    switch (iter % 4) {
    case 0:
#pragma omp critical(c0) hint(omp_sync_hint_uncontended)
      sum = mysum + sum;
      break;
    case 1:
#pragma omp critical(c1) hint(omp_sync_hint_contended)
      sum = mysum + sum;
      break;
    case 2:
#pragma omp critical(c2) hint(omp_sync_hint_nonspeculative)
      sum = mysum + sum;
      break;
    case 3:
#pragma omp critical(c3) hint(omp_sync_hint_speculative)
      sum = mysum + sum;
      break;
    default:;
    }
  }
  known_sum = 999 * 1000 / 2;
  return (known_sum == sum);
}

int main() {
  int i;
  int num_failed = 0;

  for (i = 0; i < 4 * REPETITIONS; i++) {
    if (!test_omp_critical(i)) {
      num_failed++;
    }
  }
  return num_failed;
}
