// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

#define NTIMES 100

#define ASSERT_CMP(lhs, cmp, rhs)                                              \
  if (!((lhs)cmp(rhs))) {                                                      \
    printf("Expected: (" #lhs ") " #cmp " (" #rhs "), actual: %e vs. %e", lhs, \
           rhs);                                                               \
    return EXIT_FAILURE;                                                       \
  }

int main() {
  int i;

  for (i = 0; i < NTIMES; i++) {
    double start = omp_get_wtime(), end;
    ASSERT_CMP(start, >=, 0.0);
    for (end = omp_get_wtime(); end == start; end = omp_get_wtime()) {
      ASSERT_CMP(end, >=, 0.0);
    }
    ASSERT_CMP(end, >=, 0.0);
    ASSERT_CMP(end, >, start);
  }

  return EXIT_SUCCESS;
}
