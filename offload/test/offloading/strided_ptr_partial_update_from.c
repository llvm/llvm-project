// This test checks that #pragma omp target update from(data[0:2:3]) correctly
// updates every third element (stride 3) from the device to the host, partially
// across the array using dynamically allocated memory.

// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int len = 11;
  double *data = (double *)calloc(len, sizeof(double));

#pragma omp target map(tofrom : data[0 : len])
  {
    for (int i = 0; i < len; i++)
      data[i] = i;
  }

  // Initial values
  printf("original host array values:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
  printf("\n");

#pragma omp target data map(to : data[0 : len])
  {
#pragma omp target
    for (int i = 0; i < len; i++)
      data[i] += i;

#pragma omp target update from(data[0 : 2 : 3]) // indices 0,3 only
  }

  printf("device array values after update from:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
  printf("\n");

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

  // CHECK: device array values after update from:
  // CHECK-NEXT: 0.000000
  // CHECK-NEXT: 1.000000
  // CHECK-NEXT: 2.000000
  // CHECK-NEXT: 6.000000
  // CHECK-NEXT: 4.000000
  // CHECK-NEXT: 5.000000
  // CHECK-NEXT: 6.000000
  // CHECK-NEXT: 7.000000
  // CHECK-NEXT: 8.000000
  // CHECK-NEXT: 9.000000
  // CHECK-NEXT: 10.000000

  free(data);
  return 0;
}
