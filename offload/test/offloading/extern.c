// RUN: %libomptarget-compile-generic -DVAR -c -o %t.o
// RUN: %libomptarget-compile-generic %t.o && %libomptarget-run-generic | %fcheck-generic

#ifdef VAR
int x = 1;
#else
#include <stdio.h>
#include <assert.h>
extern int x;

int main() {
  int value = 0;
#pragma omp target map(from : value)
  value = x;
  assert(value == 1);

  x = 999;
#pragma omp target update to(x)

#pragma omp target map(from : value)
  value = x;
  assert(value == 999);

  // CHECK: PASS
  printf ("PASS\n");
}
#endif
