// RUN: %libomptarget-compile-generic -DSHARED -fPIC -shared -o %t.so && %libomptarget-compile-generic %t.so && %libomptarget-run-generic 2>&1 | %fcheck-generic

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
