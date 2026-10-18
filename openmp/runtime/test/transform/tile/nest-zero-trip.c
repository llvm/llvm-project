// RUN: %libomp-compile -fopenmp-version=60 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// After tile/stripe of a loop nest, each loop variable must hold the value it
// would have without the directive (OpenMP 6.0). An inner loop variable is
// only assigned if every enclosing loop has iterations.

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

#define NEST(PRAGMA, NAME)                                                     \
  static void NAME(int n, int m) {                                             \
    int i = 77, j = 111;                                                       \
    _Pragma(PRAGMA)                                                            \
    for (i = 0; i < n; i++)                                                    \
      for (j = 0; j < m; j++)                                                  \
        ;                                                                      \
    printf(#NAME " n=%d m=%d: i=%d j=%d\n", n, m, i, j);                       \
  }

NEST("omp tile sizes(2, 3)", tile)
NEST("omp stripe sizes(2, 3)", stripe)

int main() {
  tile(0, 5);   // outer loop never runs: j untouched
  tile(6, 0);   // inner loop never runs: j gets its init value
  tile(6, 5);   // all loops run: exit values
  stripe(0, 5);
  stripe(6, 0);
  stripe(6, 5);
  return EXIT_SUCCESS;
}
#endif /* HEADER */

// CHECK:      tile n=0 m=5: i=0 j=111
// CHECK-NEXT: tile n=6 m=0: i=6 j=0
// CHECK-NEXT: tile n=6 m=5: i=6 j=5
// CHECK-NEXT: stripe n=0 m=5: i=0 j=111
// CHECK-NEXT: stripe n=6 m=0: i=6 j=0
// CHECK-NEXT: stripe n=6 m=5: i=6 j=5
