// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_DEBUG=KernelArgs %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=SEL
// RUN: env LIBOMPTARGET_INFO=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=SEL
// RUN: env LIBOMPTARGET_DEBUG=DumpTable %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=OTHER --allow-empty

#include <stdio.h>

int main() {
  int A[64];
  int S = 3;
#pragma omp target map(tofrom : A) firstprivate(S)
  {
    A[0] = S;
  }
  return 0;
}

// SEL: with {{[0-9]+}} arguments:
// OTHER-NOT: with {{[0-9]+}} arguments:
