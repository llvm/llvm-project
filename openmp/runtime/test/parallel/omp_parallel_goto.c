// RUN: %libomp-compile
// RUN: %libomp-run

// XFAIL: irbuilder

#include <stddef.h>
#include <stdio.h>
#include <omp.h>

int main() {
  int result[] = {0, 0};

#pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num();
    result[tid] += 1;
    goto cont;

  orphaned:
    result[tid] += 2;
    printf("Never executed\n");

  cont:
    result[tid] += 4;
  }

  if (result[0] == 5 && result[1] == 5) {
    printf("SUCCESS\n");
    return EXIT_SUCCESS;
  }

  printf("FAILED\n");
  return EXIT_FAILURE;
}
