// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic
// https://github.com/llvm/llvm-project/issues/182119
// UNSUPPORTED: intelgpu

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
