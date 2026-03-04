// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
// This test checks that "update to" clause in OpenMP supports strided sections.
// #pragma omp target update to(result[0:8:2]) updates every other element
// (stride 2)

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16

int main() {
  double *result = (double *)calloc(N, sizeof(double));

  // Initialize on host
  for (int i = 0; i < N; i++) {
    result[i] = i;
  }

  // Initial values
  printf("original host array values:\n");
  for (int i = 0; i < N; i++)
    printf("%f\n", result[i]);
  printf("\n");

#pragma omp target data map(from : result[0 : N])
  {
// Initialize device array to 0
#pragma omp target
    {
      for (int i = 0; i < N; i++) {
        result[i] = 0.0;
      }
    }

    // Modify host strided elements (even indices)
    for (int i = 0; i < 8; i++) {
      result[i * 2] = 100.0;
    }

// Update strided elements to device: indices 0,2,4,6,8,10,12,14
#pragma omp target update to(result[0 : 8 : 2])

#pragma omp target
    {
      for (int i = 0; i < N; i++) {
        result[i] += i;
      }
    }
  }

  printf("from target array results:\n");
  for (int i = 0; i < N; i++)
    printf("%f\n", result[i]);

  // CHECK: original host array values:
  // CHECK-NEXT: 0.000000
  // CHECK-NEXT: 1.000000
  // CHECK-NEXT: 2.000000
  // CHECK-NEXT: 3.000000
  // CHECK-NEXT: 4.000000
  // CHECK-NEXT: 5.000000
  // CHECK-NEXT: 6.000000
  // CHECK-NEXT: 7.000000
  // CHECK-NEXT: 8.000000
  // CHECK-NEXT: 9.000000
  // CHECK-NEXT: 10.000000
  // CHECK-NEXT: 11.000000
  // CHECK-NEXT: 12.000000
  // CHECK-NEXT: 13.000000
  // CHECK-NEXT: 14.000000
  // CHECK-NEXT: 15.000000

  // CHECK: from target array results:
  // CHECK-NEXT: 100.000000
  // CHECK-NEXT: 1.000000
  // CHECK-NEXT: 102.000000
  // CHECK-NEXT: 3.000000
  // CHECK-NEXT: 104.000000
  // CHECK-NEXT: 5.000000
  // CHECK-NEXT: 106.000000
  // CHECK-NEXT: 7.000000
  // CHECK-NEXT: 108.000000
  // CHECK-NEXT: 9.000000
  // CHECK-NEXT: 110.000000
  // CHECK-NEXT: 11.000000
  // CHECK-NEXT: 112.000000
  // CHECK-NEXT: 13.000000
  // CHECK-NEXT: 114.000000
  // CHECK-NEXT: 15.000000

  free(result);
  return 0;
}
