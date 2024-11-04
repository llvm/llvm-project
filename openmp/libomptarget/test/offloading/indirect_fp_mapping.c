// RUN: %libomptarget-compile-generic -fopenmp-version=51
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=51
// RUN: %libomptarget-run-generic | %fcheck-generic

#include <stdio.h>

int square(int x) { return x * x; }
#pragma omp declare target indirect to(square)

typedef int (*fp_t)(int);

int main() {
  int i = 17, r;

  fp_t fp = &square;
  // CHECK: host: &square =
  printf("host: &square = %p\n", fp);

#pragma omp target map(from : fp)
  fp = &square;
  // CHECK: device: &square = [[DEV_FP:.*]]
  printf("device: &square = %p\n", fp);

  fp_t fp1 = square;
  fp_t fp2 = 0;
#pragma omp target map(from : fp2)
  fp2 = fp1;
  // CHECK: device: fp2 = [[DEV_FP]]
  printf("device: fp2 = %p\n", fp2);

#pragma omp target map(from : r)
  { r = fp1(i); }

  // CHECK: 17*17 = 289
  printf("%i*%i = %i\n", i, i, r);
}
