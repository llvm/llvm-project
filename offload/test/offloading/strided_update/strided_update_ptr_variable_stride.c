// RUN: %libomptarget-compile-run-and-check-generic
// Tests non-contiguous array sections with variable stride on heap-allocated
// pointers.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int stride = 2;
  int len = 10;
  double *result = (double *)malloc(len * sizeof(double));

  // Initialize
  for (int i = 0; i < len; i++) {
    result[i] = 0;
  }

#pragma omp target enter data map(to : stride, len, result[0 : len])

#pragma omp target map(alloc : result[0 : len])
  {
    for (int i = 0; i < len; i++) {
      result[i] = i;
    }
  }

  // Test FROM
#pragma omp target map(alloc : result[0 : len])
  {
    for (int i = 0; i < len; i++) {
      result[i] += i * 10;
    }
  }

#pragma omp target update from(result[0 : 5 : stride])

  printf("heap ptr variable stride (from):\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", result[i]);

  // Test TO: Reset, modify host, update to device
#pragma omp target map(alloc : result[0 : len])
  {
    for (int i = 0; i < len; i++) {
      result[i] = i * 2;
    }
  }

  for (int i = 0; i < 5; i++) {
    result[i * stride] = i + 100;
  }

#pragma omp target update to(result[0 : 5 : stride])

#pragma omp target map(alloc : result[0 : len])
  {
    for (int i = 0; i < len; i++) {
      result[i] += 100;
    }
  }

#pragma omp target update from(result[0 : len])

  printf("heap ptr variable stride (to):\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", result[i]);

#pragma omp target exit data map(delete : stride, len, result[0 : len])
  free(result);
  return 0;
}

// CHECK: heap ptr variable stride (from):
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 22.000000
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 44.000000
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 66.000000
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 88.000000
// CHECK-NEXT: 0.000000
// CHECK: heap ptr variable stride (to):
// CHECK-NEXT: 200.000000
// CHECK-NEXT: 102.000000
// CHECK-NEXT: 201.000000
// CHECK-NEXT: 106.000000
// CHECK-NEXT: 202.000000
// CHECK-NEXT: 110.000000
// CHECK-NEXT: 203.000000
// CHECK-NEXT: 114.000000
// CHECK-NEXT: 204.000000
// CHECK-NEXT: 118.000000
