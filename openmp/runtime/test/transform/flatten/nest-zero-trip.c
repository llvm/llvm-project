// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// After flatten of a loop nest, each loop variable must hold the value it
// would have without the directive. An inner loop variable is only assigned
// if every enclosing loop has iterations.

#include <stdlib.h>
#include <stdio.h>

static void flatten(int n, int m) {
  int i = 77, j = 111;
#pragma omp flatten
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      ;
  printf("flatten n=%d m=%d: i=%d j=%d\n", n, m, i, j);
}

int main() {
  // Outer loop never runs: j untouched.
  flatten(0, 5);
  // Inner loop never runs: j gets its init value.
  flatten(6, 0);
  // All loops run: exit values.
  flatten(6, 5);
  return EXIT_SUCCESS;
}

// CHECK:      flatten n=0 m=5: i=0 j=111
// CHECK-NEXT: flatten n=6 m=0: i=6 j=0
// CHECK-NEXT: flatten n=6 m=5: i=6 j=5
