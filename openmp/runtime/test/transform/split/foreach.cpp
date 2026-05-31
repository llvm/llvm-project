// RUN: %libomp-cxx-compile-and-run | FileCheck %s --match-full-lines

#include <cstdlib>
#include <cstdio>
#include <vector>

int main() {
  std::vector<int> v = {10, 20, 30, 40, 50, 60};
  printf("do\n");
#pragma omp split counts(2, omp_fill)
  for (int x : v)
    printf("x=%d\n", x);
  printf("done\n");
  return EXIT_SUCCESS;
}

// CHECK:      do
// CHECK-NEXT: x=10
// CHECK-NEXT: x=20
// CHECK-NEXT: x=30
// CHECK-NEXT: x=40
// CHECK-NEXT: x=50
// CHECK-NEXT: x=60
// CHECK-NEXT: done
