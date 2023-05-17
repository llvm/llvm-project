// RUN: %libomptarget-compile-run-and-check-generic

#include <assert.h>
#include <stdio.h>

int main() {
  int i = 1;
#pragma omp target
  assert(i > 0);

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
