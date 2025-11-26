// This test checks that "update from" clause in OpenMP is supported when the
// elements are updated in a non-contiguous manner. This test checks that
// #pragma omp target update from(data[0:4:2]) correctly updates only every
// other element (stride 2) from the device to the host

// RUN: %libomptarget-compile-run-and-check-generic
#include <omp.h>
#include <stdio.h>

int main() {
  int len = 8;
  double data[len];
#pragma omp target map(tofrom : len, data[0 : len])
  {
    for (int i = 0; i < len; i++) {
      data[i] = i;
    }
  }
  // Initial values
  printf("original host array values:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
  printf("\n");

#pragma omp target data map(to : len, data[0 : len])
  {
// Modify arrays on device
#pragma omp target
    for (int i = 0; i < len; i++) {
      data[i] += i;
    }

#pragma omp target update from(data[0 : 4 : 2])
  }
  // CHECK: 0.000000
  // CHECK: 1.000000
  // CHECK: 4.000000
  // CHECK: 3.000000
  // CHECK: 8.000000
  // CHECK: 5.000000
  // CHECK: 12.000000
  // CHECK: 7.000000
  // CHECK-NOT: 2.000000
  // CHECK-NOT: 6.000000
  // CHECK-NOT: 10.000000
  // CHECK-NOT: 14.000000

  printf("from target array results:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
  printf("\n");

  return 0;
}
