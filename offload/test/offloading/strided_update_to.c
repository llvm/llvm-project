// This test checks that "update to" clause in OpenMP is supported when the
// elements are updated in a non-contiguous manner. This test checks that
// #pragma omp target update to(data[0:4:2]) correctly updates only every
// other element (stride 2) from the host to the device

// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu

#include <omp.h>
#include <stdio.h>

int main() {
  int len = 8;
  double data[len];

  // Initialize on host
  for (int i = 0; i < len; i++) {
    data[i] = i;
  }

  // Initial values
  printf("original host array values:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
  printf("\n");

#pragma omp target data map(tofrom : len, data[0 : len])
  {
    // Initialize device to 20
#pragma omp target
    for (int i = 0; i < len; i++) {
      data[i] = 20.0;
    }

    // Modify host for strided elements
    data[0] = 10.0;
    data[2] = 10.0;
    data[4] = 10.0;
    data[6] = 10.0;

#pragma omp target update to(data[0 : 4 : 2])

    // Verify on device by adding 5
#pragma omp target
    for (int i = 0; i < len; i++) {
      data[i] += 5.0;
    }
  }

  // CHECK: 0.000000
  // CHECK: 1.000000
  // CHECK: 2.000000
  // CHECK: 3.000000
  // CHECK: 4.000000
  // CHECK: 5.000000
  // CHECK: 6.000000
  // CHECK: 7.000000

  printf("device array values after update to:\n");
  for (int i = 0; i < len; i++)
    printf("%f\n", data[i]);
  printf("\n");

  // CHECK: 15.000000
  // CHECK: 25.000000
  // CHECK: 15.000000
  // CHECK: 25.000000
  // CHECK: 15.000000
  // CHECK: 25.000000
  // CHECK: 15.000000
  // CHECK: 25.000000

  return 0;
}
