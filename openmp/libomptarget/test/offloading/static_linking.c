// RUN: %libomptarget-compile-generic -DLIBRARY -c -o %t.o
// RUN: llvm-ar rcs %t.a %t.o
// RUN: %libomptarget-compile-generic %t.a && %libomptarget-run-generic 2>&1 | %fcheck-generic

// REQUIRES: nvptx64-nvidia-cuda-newDriver
// REQUIRES: amdgcn-amd-amdhsa-newDriver

#ifdef LIBRARY
int x = 42;
#pragma omp declare target(x)

int foo() {
  int value;
#pragma omp target map(from : value)
  value = x;
  return value;
}
#else
#include <stdio.h>
int foo();

int main() {
  int x = foo();

  // CHECK: PASS
  if (x == 42)
    printf("PASS\n");
}
#endif
