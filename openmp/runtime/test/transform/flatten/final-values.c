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

  // An empty outer loop still runs its init, but never enters the body, so the
  // inner init does not run and 'j' keeps the value from before the nest.
  i = -1;
  j = -1;
#pragma omp flatten
  for (i = 0; i < Zero; ++i)
    for (j = 0; j < 3; ++j)
      ;

  printf("empty-outer-i=%d empty-outer-j=%d\n", i, j);

  // The middle loop is empty, so the innermost init never runs.
  int k = -1;
  i = -1;
  j = -1;
#pragma omp flatten depth(3)
  for (i = 0; i < 2; ++i)
    for (j = 0; j < Zero; ++j)
      for (k = 0; k < 3; ++k)
        ;

  printf("mid-empty-i=%d mid-empty-j=%d mid-empty-k=%d\n", i, j, k);

  // Outer init is a DeclStmt (always run). The inner assignment is still
  // skipped when that outer loop is empty.
  j = -1;
#pragma omp flatten
  for (int ii = 0; ii < Zero; ++ii)
    for (j = 0; j < 3; ++j)
      ;

  printf("decl-outer-j=%d\n", j);

  return EXIT_SUCCESS;
}

// CHECK:      i=2 j=3
// CHECK-NEXT: descending-i=-1 descending-j=1
// CHECK-NEXT: empty-inner-i=3 empty-inner-j=0
// CHECK-NEXT: empty-outer-i=0 empty-outer-j=-1
// CHECK-NEXT: mid-empty-i=2 mid-empty-j=0 mid-empty-k=-1
// CHECK-NEXT: decl-outer-j=-1
