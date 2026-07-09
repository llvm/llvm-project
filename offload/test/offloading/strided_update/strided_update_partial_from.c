// This test checks that #pragma omp target update from(data[0:4:3]) correctly
// updates every third element (stride 3) from the device to the host, partially
// across the array

// RUN: %libomptarget-compile-run-and-check-generic
#include <omp.h>
#include <stdio.h>

int main() {
  int len = 11;
  double data[len];

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
// Modify arrays on device
#pragma omp target
    for (int i = 0; i < len; i++)
      data[i] += i;

#pragma omp target update from(data[0 : 4 : 3]) // indices 0,3,6,9
  }

  printf("device array values after update from:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
  printf("\n");

  // CHECK: 0.000000
  // CHECK: 1.000000
  // CHECK: 2.000000
  // CHECK: 3.000000
  // CHECK: 4.000000
  // CHECK: 5.000000
  // CHECK: 6.000000
  // CHECK: 7.000000
  // CHECK: 8.000000
  // CHECK: 9.000000
  // CHECK: 10.000000

  // CHECK: 0.000000
  // CHECK: 1.000000
  // CHECK: 2.000000
  // CHECK: 6.000000
  // CHECK: 4.000000
  // CHECK: 5.000000
  // CHECK: 12.000000
  // CHECK: 7.000000
  // CHECK: 8.000000
  // CHECK: 18.000000
  // CHECK: 10.000000
}
