// RUN: %libomp-cxx-compile-and-run | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdio>

int main() {
  printf("do\n");
#pragma omp parallel for collapse(4) num_threads(1)
  for (int i = 0; i < 3; ++i)
#pragma omp interchange
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          printf("i=%d j=%d k=%d l=%d\n", i, j, k, l);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: i=0 j=0 k=0 l=0
// CHECK-NEXT: i=0 j=0 k=0 l=1
// CHECK-NEXT: i=0 j=0 k=0 l=2
// CHECK-NEXT: i=0 j=1 k=0 l=0
// CHECK-NEXT: i=0 j=1 k=0 l=1
// CHECK-NEXT: i=0 j=1 k=0 l=2
// CHECK-NEXT: i=0 j=2 k=0 l=0
// CHECK-NEXT: i=0 j=2 k=0 l=1
// CHECK-NEXT: i=0 j=2 k=0 l=2
// CHECK-NEXT: i=0 j=0 k=1 l=0
// CHECK-NEXT: i=0 j=0 k=1 l=1
// CHECK-NEXT: i=0 j=0 k=1 l=2
// CHECK-NEXT: i=0 j=1 k=1 l=0
// CHECK-NEXT: i=0 j=1 k=1 l=1
// CHECK-NEXT: i=0 j=1 k=1 l=2
// CHECK-NEXT: i=0 j=2 k=1 l=0
// CHECK-NEXT: i=0 j=2 k=1 l=1
// CHECK-NEXT: i=0 j=2 k=1 l=2
// CHECK-NEXT: i=0 j=0 k=2 l=0
// CHECK-NEXT: i=0 j=0 k=2 l=1
// CHECK-NEXT: i=0 j=0 k=2 l=2
// CHECK-NEXT: i=0 j=1 k=2 l=0
// CHECK-NEXT: i=0 j=1 k=2 l=1
// CHECK-NEXT: i=0 j=1 k=2 l=2
// CHECK-NEXT: i=0 j=2 k=2 l=0
// CHECK-NEXT: i=0 j=2 k=2 l=1
// CHECK-NEXT: i=0 j=2 k=2 l=2
// CHECK-NEXT: i=1 j=0 k=0 l=0
// CHECK-NEXT: i=1 j=0 k=0 l=1
// CHECK-NEXT: i=1 j=0 k=0 l=2
// CHECK-NEXT: i=1 j=1 k=0 l=0
// CHECK-NEXT: i=1 j=1 k=0 l=1
// CHECK-NEXT: i=1 j=1 k=0 l=2
// CHECK-NEXT: i=1 j=2 k=0 l=0
// CHECK-NEXT: i=1 j=2 k=0 l=1
// CHECK-NEXT: i=1 j=2 k=0 l=2
// CHECK-NEXT: i=1 j=0 k=1 l=0
// CHECK-NEXT: i=1 j=0 k=1 l=1
// CHECK-NEXT: i=1 j=0 k=1 l=2
// CHECK-NEXT: i=1 j=1 k=1 l=0
// CHECK-NEXT: i=1 j=1 k=1 l=1
// CHECK-NEXT: i=1 j=1 k=1 l=2
// CHECK-NEXT: i=1 j=2 k=1 l=0
// CHECK-NEXT: i=1 j=2 k=1 l=1
// CHECK-NEXT: i=1 j=2 k=1 l=2
// CHECK-NEXT: i=1 j=0 k=2 l=0
// CHECK-NEXT: i=1 j=0 k=2 l=1
// CHECK-NEXT: i=1 j=0 k=2 l=2
// CHECK-NEXT: i=1 j=1 k=2 l=0
// CHECK-NEXT: i=1 j=1 k=2 l=1
// CHECK-NEXT: i=1 j=1 k=2 l=2
// CHECK-NEXT: i=1 j=2 k=2 l=0
// CHECK-NEXT: i=1 j=2 k=2 l=1
// CHECK-NEXT: i=1 j=2 k=2 l=2
// CHECK-NEXT: i=2 j=0 k=0 l=0
// CHECK-NEXT: i=2 j=0 k=0 l=1
// CHECK-NEXT: i=2 j=0 k=0 l=2
// CHECK-NEXT: i=2 j=1 k=0 l=0
// CHECK-NEXT: i=2 j=1 k=0 l=1
// CHECK-NEXT: i=2 j=1 k=0 l=2
// CHECK-NEXT: i=2 j=2 k=0 l=0
// CHECK-NEXT: i=2 j=2 k=0 l=1
// CHECK-NEXT: i=2 j=2 k=0 l=2
// CHECK-NEXT: i=2 j=0 k=1 l=0
// CHECK-NEXT: i=2 j=0 k=1 l=1
// CHECK-NEXT: i=2 j=0 k=1 l=2
// CHECK-NEXT: i=2 j=1 k=1 l=0
// CHECK-NEXT: i=2 j=1 k=1 l=1
// CHECK-NEXT: i=2 j=1 k=1 l=2
// CHECK-NEXT: i=2 j=2 k=1 l=0
// CHECK-NEXT: i=2 j=2 k=1 l=1
// CHECK-NEXT: i=2 j=2 k=1 l=2
// CHECK-NEXT: i=2 j=0 k=2 l=0
// CHECK-NEXT: i=2 j=0 k=2 l=1
// CHECK-NEXT: i=2 j=0 k=2 l=2
// CHECK-NEXT: i=2 j=1 k=2 l=0
// CHECK-NEXT: i=2 j=1 k=2 l=1
// CHECK-NEXT: i=2 j=1 k=2 l=2
// CHECK-NEXT: i=2 j=2 k=2 l=0
// CHECK-NEXT: i=2 j=2 k=2 l=1
// CHECK-NEXT: i=2 j=2 k=2 l=2
// CHECK-NEXT: done
