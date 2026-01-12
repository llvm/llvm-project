// RUN: %libomptarget-compile-run-and-check-generic
// Tests non-contiguous array sections with expression-based count on
// heap-allocated pointer arrays with both FROM and TO directives.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int len = 10;
  double *result = (double *)malloc(len * sizeof(double));

  // Initialize host array to zero
  for (int i = 0; i < len; i++) {
    result[i] = 0;
  }

  // Initialize on device
#pragma omp target enter data map(to : len, result[0 : len])

#pragma omp target map(alloc : result[0 : len])
  {
    for (int i = 0; i < len; i++) {
      result[i] = i;
    }
  }

  // Test FROM: Modify on device, then update from device
#pragma omp target map(alloc : result[0 : len])
  {
    for (int i = 0; i < len; i++) {
      result[i] += i * 10;
    }
  }

  // Update from device with expression-based count: len/2 elements
#pragma omp target update from(result[0 : len / 2 : 2])

  printf("heap ptr count expression (from):\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", result[i]);

  // Test TO: Reset, modify host, update to device
#pragma omp target map(alloc : result[0 : len])
  {
    for (int i = 0; i < len; i++) {
      result[i] = i * 2;
    }
  }

  // Modify host data
  for (int i = 0; i < len / 2; i++) {
    result[i * 2] = i + 100;
  }

  // Update to device with expression-based count
#pragma omp target update to(result[0 : len / 2 : 2])

  // Read back full array
#pragma omp target map(alloc : result[0 : len])
  {
    for (int i = 0; i < len; i++) {
      result[i] += 100;
    }
  }

#pragma omp target update from(result[0 : len])

  printf("heap ptr count expression (to):\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", result[i]);

#pragma omp target exit data map(delete : len, result[0 : len])
  free(result);
  return 0;
}

// CHECK: heap ptr count expression (from):
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
// CHECK: heap ptr count expression (to):
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
