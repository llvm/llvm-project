// RUN: %libomp-compile
// RUN: %libomp-run | FileCheck %s --check-prefix OMP-CHECK

#include <stdio.h>

void foo(int x) {
  printf("foo");
  return;
}

int main() {
  int x = 4;
  // should call foo()
  if (x % 2 == 0)
#pragma omp nothing
    foo(x);

  // should not call foo()
  x = 3;
  if (x % 2 == 0)
#pragma omp nothing
    foo(x);

  // OMP-CHECK: foo
  // OMP-CHECK-NOT: foo
  return 0;
}