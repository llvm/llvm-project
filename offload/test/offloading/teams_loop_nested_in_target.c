// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

// A 'teams loop' nested in a 'target' region is emitted as 'teams distribute'
// and must not be executed in SPMD mode.

#include <stdio.h>

#define N (1 << 20)

int Y[N];

int main() {
  for (int I = 0; I < N; ++I)
    Y[I] = 1;

  long Sum = 0;
#pragma omp target map(to : Y) map(tofrom : Sum)
#pragma omp teams loop reduction(+ : Sum)
  for (int I = 0; I < N; ++I)
    Sum += Y[I];
  // CHECK: reduction: 1048576
  printf("reduction: %ld\n", Sum);

#pragma omp target map(tofrom : Y)
#pragma omp teams loop
  for (int I = 0; I < N; ++I)
    Y[I] += 1;
  Sum = 0;
  for (int I = 0; I < N; ++I)
    Sum += Y[I];
  // CHECK: increments: 2097152
  printf("increments: %ld\n", Sum);

  return 0;
}
