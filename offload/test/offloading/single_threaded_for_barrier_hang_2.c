// RUN: %libomptarget-compile-run-and-check-generic

// FIXME: https://github.com/llvm/llvm-project/issues/161265
// UNSUPPORTED: gpu

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
