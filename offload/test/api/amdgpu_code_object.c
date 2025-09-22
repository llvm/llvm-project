// RUN: %libomptarget-compile-amdgcn-amd-amdhsa -Xclang \
// RUN:   -mcode-object-version=5
// RUN:   %libomptarget-run-amdgcn-amd-amdhsa | %fcheck-amdgcn-amd-amdhsa

// REQUIRES: amdgcn-amd-amdhsa

#include <stdio.h>

// Test to make sure we can build and run with the previous COV.
int main() {
#pragma omp target
  ;

  // CHECK: PASS
  printf("PASS\n");
}
