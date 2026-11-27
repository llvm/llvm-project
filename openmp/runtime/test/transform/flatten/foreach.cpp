// RUN: %libomp-cxx-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdio>
#include <vector>

int main() {
  printf("do\n");
  std::vector<int> outer{10, 20};
  std::vector<int> inner{1, 2, 3};
#pragma omp flatten
  for (int a : outer)
    for (int b : inner)
      printf("a=%d b=%d\n", a, b);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: a=10 b=1
// CHECK-NEXT: a=10 b=2
// CHECK-NEXT: a=10 b=3
// CHECK-NEXT: a=20 b=1
// CHECK-NEXT: a=20 b=2
// CHECK-NEXT: a=20 b=3
// CHECK-NEXT: done
