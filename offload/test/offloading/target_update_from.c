// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
// This test checks that "update from" clause in OpenMP supports strided
// sections. #pragma omp target update from(result[0:N/2:2]) updates every other
// element from device
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 32

int main() {
  double *result = (double *)calloc(N, sizeof(double));

  printf("initial host array values:\n");
  for (int i = 0; i < N; i++)
    printf("%f\n", result[i]);
  printf("\n");

#pragma omp target data map(to : result[0 : N])
  {
#pragma omp target map(alloc : result[0 : N])
    for (int i = 0; i < N; i++)
      result[i] += i;

    // Update strided elements from device: even indices 0,2,4,...,30
#pragma omp target update from(result[0 : 16 : 2])
  }

  printf("after target update from (even indices up to 30 updated):\n");
  for (int i = 0; i < N; i++)
    printf("%f\n", result[i]);
  printf("\n");

  // Expected: even indices i, odd indices 0
  // CHECK: after target update from
  // CHECK: 0.000000
  // CHECK: 0.000000
  // CHECK: 2.000000
  // CHECK: 0.000000
  // CHECK: 4.000000
  // CHECK: 0.000000
  // CHECK: 6.000000
  // CHECK: 0.000000
  // CHECK: 8.000000
  // CHECK: 0.000000
  // CHECK: 10.000000
  // CHECK: 0.000000
  // CHECK: 12.000000
  // CHECK: 0.000000
  // CHECK: 14.000000
  // CHECK: 0.000000
  // CHECK: 16.000000
  // CHECK: 0.000000
  // CHECK: 18.000000
  // CHECK: 0.000000
  // CHECK: 20.000000
  // CHECK: 0.000000
  // CHECK: 22.000000
  // CHECK: 0.000000
  // CHECK: 24.000000
  // CHECK: 0.000000
  // CHECK: 26.000000
  // CHECK: 0.000000
  // CHECK: 28.000000
  // CHECK: 0.000000
  // CHECK: 30.000000
  // CHECK: 0.000000
  // CHECK-NOT: 1.000000
  // CHECK-NOT: 3.000000
  // CHECK-NOT: 31.000000

  free(result);
  return 0;
}
