// RUN: %libomptarget-compile-generic -DA -c -o %t-a.o
// RUN: %libomptarget-compile-generic -DB -c -o %t-b.o
// RUN: %libomptarget-compile-generic %t-a.o %t-b.o && \
// RUN:   %libomptarget-run-generic | %fcheck-generic

#if defined(A)
__attribute__((weak)) int x = 999;
#pragma omp declare target to(x)
#elif defined(B)
int x = 42;
#pragma omp declare target to(x)
__attribute__((weak)) int y = 42;
#pragma omp declare target to(y)
#else

#include <stdio.h>

extern int x;
#pragma omp declare target to(x)
extern int y;
#pragma omp declare target to(y)

int main() {
  x = 0;

#pragma omp target update from(x)
#pragma omp target update from(y)

  // CHECK: PASS
  if (x == 42 && y == 42)
    printf("PASS\n");
}
#endif
