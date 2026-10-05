// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int i = -1;
  int j = -1;

#pragma omp flatten
  for (i = 0; i < 2; ++i)
    for (j = 0; j < 3; ++j)
      if (i == 1 && j == 2)
        continue;

  printf("i=%d j=%d\n", i, j);

#pragma omp flatten
  for (i = 5; i > 0; i -= 2)
    for (j = 7; j > 1; j -= 3)
      ;

  printf("descending-i=%d descending-j=%d\n", i, j);

  // An empty inner loop executes no body, but the outer loop still runs to
  // completion, so 'i' reaches its final value while 'j' keeps the value from
  // its initialization.
  int Zero = 0;
#pragma omp flatten
  for (i = 0; i < 3; ++i)
    for (j = 0; j < Zero; ++j)
      ;

  printf("empty-inner-i=%d empty-inner-j=%d\n", i, j);
  return EXIT_SUCCESS;
}

// CHECK:      i=2 j=3
// CHECK-NEXT: descending-i=-1 descending-j=1
// CHECK-NEXT: empty-inner-i=3 empty-inner-j=0
