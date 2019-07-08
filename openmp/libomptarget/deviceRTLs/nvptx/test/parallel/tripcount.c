// RUN: %compile-run-and-check

#include <omp.h>
#include <stdio.h>

int main() {
  int res = 0;

#pragma omp parallel num_threads(2) reduction(+:res)
  {
    int tid = omp_get_thread_num();
#pragma omp target teams distribute reduction(+:res)
    for (int i = tid; i < 2; i++)
      ++res;
  }
  // The first thread makes 2 iterations, the second - 1. Expected result of the
  // reduction res is 3.

  // CHECK: res = 3.
  printf("res = %d.\n", res);
  return 0;
}
