// RUN: %libomptarget-compile-run-and-check-generic

#include <stdio.h>

int main() {
  int x = 111;
#pragma omp target data map(alloc : x) map(tofrom : x) map(alloc : x)
  {
#pragma omp target map(present, alloc : x)
    {
      printf("%d\n", x); // CHECK: 111
      x = x + 111;
    }
  }

  printf("%d\n", x); // CHECK: 222
}
