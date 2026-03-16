// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu

#include <stdio.h>

int main() {
  int x = 111;
#pragma omp target map(alloc : x) map(tofrom : x) map(alloc : x)
  {
    x = x + 111;
  }

  printf("After tgt: %d\n", x); // CHECK: After tgt: 222
}
