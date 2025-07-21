// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

#include <omp.h>
#include <stdio.h>

int main() {
  int b = 0;

#pragma omp target map(tofrom : b)
  for (int i = 1; i <= 10; ++i) {
#pragma omp parallel num_threads(10) reduction(+ : b)
#pragma omp for
    for (int k = 0; k < 10; ++k)
      ++b;
  }

  // CHECK: b: 100
  printf("b: %i\n", b);
  return 0;
}
