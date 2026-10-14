// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

// Cross-team reductions in kernels without a parallel region. With
// optimizations, such kernels are executed with only one thread per team
// ("fake"-SPMD mode), which the cross-team reduction has to take into account.

#include <stdio.h>

#define N (1 << 20)

int Y[N];

#pragma omp declare target
int getY(int *A, int I) { return A[I]; }
#pragma omp end declare target

int main(void) {
  for (int I = 0; I < N; ++I)
    Y[I] = 1;

  long Sum = 0;
#pragma omp target teams distribute reduction(+ : Sum) map(to : Y)
  for (int I = 0; I < N; ++I)
    Sum += Y[I];
  // CHECK: teams distribute: 1048576
  printf("teams distribute: %ld\n", Sum);

  // The call to getY prevents emitting 'loop' as 'distribute parallel for'.
  Sum = 0;
#pragma omp target teams loop reduction(+ : Sum) map(to : Y)
  for (int I = 0; I < N; ++I)
    Sum += getY(Y, I);
  // CHECK: teams loop: 1048576
  printf("teams loop: %ld\n", Sum);

  Sum = 0;
#pragma omp target map(to : Y) map(tofrom : Sum)
#pragma omp teams distribute reduction(+ : Sum)
  for (int I = 0; I < N; ++I)
    Sum += Y[I];
  // CHECK: target + teams distribute: 1048576
  printf("target + teams distribute: %ld\n", Sum);

  return 0;
}
