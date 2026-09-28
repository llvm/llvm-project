// clang-format off
// RUN: %libomptarget-compile-generic -DSHARED -fPIC -shared -o %t.so && \
// RUN: %libomptarget-compile-generic %t.so -o %t && \
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -DSHARED -fPIC -shared -o %t.so && \
// RUN: %libomptarget-compileopt-generic %t.so -o %t && \
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
//
// REQUIRES: gpu
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// clang-format on

// Checks that a target region in the main program can access a declare target
// variable that is defined in a shared library.

#ifdef SHARED
#pragma omp declare target
int x = 42;
#pragma omp end declare target
#else
#include <assert.h>
#include <stdio.h>

#pragma omp declare target
extern int x;
#pragma omp end declare target

int main() {
  int value = 0;
#pragma omp target map(from : value)
  value = x;
  assert(value == 42);

  x = 999;
#pragma omp target update to(x)

#pragma omp target map(from : value)
  value = x;
  assert(value == 999);

#pragma omp target
  x++;
#pragma omp target update from(x)
  assert(x == 1000);

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
#endif
