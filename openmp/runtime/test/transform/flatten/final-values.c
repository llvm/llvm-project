// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int I = -1;
  int J = -1;

#pragma omp flatten
  for (I = 0; I < 2; ++I)
    for (J = 0; J < 3; ++J)
      if (I == 1 && J == 2)
        continue;

  printf("i=%d j=%d\n", I, J);

#pragma omp flatten
  for (I = 5; I > 0; I -= 2)
    for (J = 7; J > 1; J -= 3)
      ;

  printf("descending-i=%d descending-j=%d\n", I, J);

  // An empty inner loop executes no body, but the outer loop still runs to
  // completion, so 'I' reaches its final value while 'J' keeps the value from
  // its initialization.
  int Zero = 0;
#pragma omp flatten
  for (I = 0; I < 3; ++I)
    for (J = 0; J < Zero; ++J)
      ;

  printf("empty-inner-i=%d empty-inner-j=%d\n", I, J);
  return EXIT_SUCCESS;
}

// CHECK:      i=2 j=3
// CHECK-NEXT: descending-i=-1 descending-j=1
// CHECK-NEXT: empty-inner-i=3 empty-inner-j=0
