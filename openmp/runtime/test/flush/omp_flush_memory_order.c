// RUN: %libomp-compile-and-run
// REQUIRES: clang

// Message passing between two threads using flush directives with
// memory-order clauses, which clang lowers to __kmpc_flush_explicit. This is
// a smoke test of that entry point: on a strongly ordered host such as x86 it
// passes even if the fences are too weak.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
  for (int n = 0; n < 1000; n++) {
    int x = 0, y = 0, fails = 0;
#pragma omp parallel num_threads(2) reduction(+ : fails)
    {
      if (omp_get_thread_num() == 0) {
        x = 10;
#pragma omp flush release
#pragma omp atomic write
        y = 1;
      } else {
        int tmp = 0;
        while (tmp == 0) {
#pragma omp atomic read
          tmp = y;
        }
#pragma omp flush acquire
        if (x != 10)
          fails++;
      }
    }
    if (fails) {
      printf("failed\n");
      return EXIT_FAILURE;
    }
  }
  printf("passed\n");
  return EXIT_SUCCESS;
}
