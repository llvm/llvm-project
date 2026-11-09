// RUN: %libomp-compile -fopenmp-version=60 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines
#include <stdio.h>
#include <stdlib.h>

static void interchange(int n, int m) {
  int i = 77, j = 111;
#pragma omp interchange permutation(2, 1)
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      ;
  printf("interchange n=%d m=%d: i=%d j=%d\n", n, m, i, j);
}

int main() {
  interchange(0, 5);
  interchange(6, 0);
  interchange(6, 5);
  return EXIT_SUCCESS;
}
// CHECK:      interchange n=0 m=5: i=0 j=111
// CHECK-NEXT: interchange n=6 m=0: i=6 j=0
// CHECK-NEXT: interchange n=6 m=5: i=6 j=5
