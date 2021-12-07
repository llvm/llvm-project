// RUN: %libomp-compile
// RUN: %libomp-run

#include <stddef.h>
#include <stdio.h>
#include <omp.h>

int main() {
  int result[2] = {0, 0};

#pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num();
#pragma omp for schedule(static)
    for (int i = 0; i < 6; i += 1)
      result[tid] += 1 << i;
  }

  if (result[0] == 1 + 2 + 4 && result[1] == 8 + 16 + 32) {
    printf("SUCCESS\n");
    return EXIT_SUCCESS;
  }
  printf("FAILED\n");
  return EXIT_FAILURE;
}
