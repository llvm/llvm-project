// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic
// UNSUPPORTED: intelgpu

#include <stdio.h>

int main(void) {
  _Float16 sum = 0;
#pragma omp target teams distribute parallel for map(tofrom : sum)             \
    reduction(+ : sum)
  for (int i = 0; i < 1024; ++i)
    sum += (_Float16)1;

  _Float16 maximum = 0;
#pragma omp target teams distribute parallel for map(tofrom : maximum)         \
    reduction(max : maximum)
  for (int i = 0; i < 1024; ++i) {
    _Float16 v = (_Float16)(i % 10);
    maximum = v > maximum ? v : maximum;
  }

  // Control: 'short' is also 2 bytes but takes the integer path, which was
  // never broken. It must stay correct.
  short int_sum = 0;
#pragma omp target teams distribute parallel for map(tofrom : int_sum)         \
    reduction(+ : int_sum)
  for (int i = 0; i < 1024; ++i)
    int_sum += 1;

  // CHECK: sum = 1024
  // CHECK: maximum = 9
  // CHECK: int_sum = 1024
  printf("sum = %g\n", (double)sum);
  printf("maximum = %g\n", (double)maximum);
  printf("int_sum = %d\n", int_sum);
  return 0;
}
