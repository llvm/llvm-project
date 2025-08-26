// RUN: %libomptarget-compile-run-and-check-generic
// FIXME: This fails with optimization enabled and prints b: 0
// FIXME: RUN: %libomptarget-compileopt-run-and-check-generic

#include <omp.h>
#include <stdio.h>

int main() {
  int b = 0;

#pragma omp target map(tofrom : b) thread_limit(256)
  for (int i = 1; i <= 1; ++i) {
#pragma omp parallel num_threads(64) reduction(+ : b)
#pragma omp parallel num_threads(10) reduction(+ : b)
#pragma omp for
    for (int k = 0; k < 10; ++k)
      ++b;
  }

  // CHECK: b: 640
  printf("b: %i\n", b);
  return 0;
}
