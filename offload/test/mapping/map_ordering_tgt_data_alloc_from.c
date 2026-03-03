// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu

#include <stdio.h>

int main() {
  int x = 111;
#pragma omp target data map(alloc : x) map(from : x) map(alloc : x)
  {
#pragma omp target map(present, alloc : x)
    x = 222;
  }

  printf("After tgt data: %d\n", x); // CHECK: After tgt data: 222
}
