// Check that a target exit data directive behaves correctly when the runtime
// has not yet been initialized.

// RUN: %libomptarget-compile-run-and-check-generic

#include <stdio.h>

int main() {
  // CHECK: x = 98
  int x = 98;
#pragma omp target exit data map(from : x)
  printf("x = %d\n", x);
  return 0;
}
