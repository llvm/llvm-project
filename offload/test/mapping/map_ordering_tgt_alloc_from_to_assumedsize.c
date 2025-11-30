// RUN: %libomptarget-compile-generic -fopenmp-version=60

// The test ensures that even though the from/to maps are on
// a list-item that looks different from the "alloc" list-items
// on assumed-size arrays, that are encountered before the to/from
// entries at run time, the transfers still kick in.

#include <omp.h>
#include <stdio.h>

int main() {
  int x[10];
  x[1] = 111;

  int *xp1, *xp2;
  xp1 = xp2 = &x[0];

#pragma omp target map(alloc : x[ : ]) map(from : xp1[1]) map(to : xp1[1])     \
    map(alloc : xp2[ : ])
  {
    printf("%d\n", x[1]); // CHECK: 111
    x[1] = x[1] + 111;
  }

  printf("%d\n", x[1]); // CHECK: 222
}
