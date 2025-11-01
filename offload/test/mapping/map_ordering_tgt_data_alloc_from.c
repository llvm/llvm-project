// RUN: %libomptarget-compile-run-and-check-generic

#include <stdio.h>

int main() {
  int x = 111;
#pragma omp target data map(alloc : x) map(from : x) map(alloc : x)
  {
#pragma omp target map(present, alloc : x)
    x = 222;
  }

  printf("%d\n", x); // CHECK: 222
}
