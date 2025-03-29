// RUN: %libomptarget-compile-generic -DSHARED -fPIC -shared -o %t.so && \
// RUN: %libomptarget-compile-generic %t.so && %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -DSHARED -fPIC -shared -o %t.so && \
// RUN: %libomptarget-compileopt-generic %t.so && %libomptarget-run-generic 2>&1 | %fcheck-generic
//
// REQUIRES: gpu

#ifdef SHARED
void foo() {}
#else
#include <stdio.h>
int main() {
#pragma omp target
  ;
  // CHECK: DONE.
  printf("%s\n", "DONE.");
  return 0;
}
#endif
