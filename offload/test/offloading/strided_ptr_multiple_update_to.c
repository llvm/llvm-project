// This test checks that #pragma omp target update to(data1[0:6:2],
// data2[0:4:3]) correctly updates strided sections covering full arrays
// from the host to the device using dynamically allocated memory.

// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int len = 12;
  double *data1 = (double *)calloc(len, sizeof(double));
  double *data2 = (double *)calloc(len, sizeof(double));

  // Initialize host arrays
  for (int i = 0; i < len; i++) {
    data1[i] = i;
    data2[i] = i * 10;
  }

  printf("original host array values:\n");
  printf("data1:\n");
  for (int i = 0; i < len; i++)
    printf("%.1f\n", data1[i]);
  printf("data2:\n");
  for (int i = 0; i < len; i++)
    printf("%.1f\n", data2[i]);

#pragma omp target data map(tofrom : data1[0 : len], data2[0 : len])
  {
// Initialize device arrays to 20
#pragma omp target
    {
      for (int i = 0; i < len; i++) {
        data1[i] = 20.0;
        data2[i] = 20.0;
      }
    }

    // data1: even indices
    for (int i = 0; i < 6; i++) {
      data1[i * 2] = 10.0;
    }
    // data2: every 3rd index
    for (int i = 0; i < 4; i++) {
      data2[i * 3] = 10.0;
    }

// data1[0:6:2] updates all even indices: 0,2,4,6,8,10 (6 elements)
// data2[0:4:3] updates every 3rd: 0,3,6,9 (4 elements)
#pragma omp target update to(data1[0 : 6 : 2], data2[0 : 4 : 3])

// Verify on device by adding 5
#pragma omp target
    {
      for (int i = 0; i < len; i++) {
        data1[i] += 5.0;
        data2[i] += 5.0;
      }
    }
  }

  printf("device array values after update to:\n");
  printf("data1:\n");
  for (int i = 0; i < len; i++)
    printf("%.1f\n", data1[i]);
  printf("data2:\n");
  for (int i = 0; i < len; i++)
    printf("%.1f\n", data2[i]);

  // CHECK: original host array values:
  // CHECK-NEXT: data1:
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 1.0
  // CHECK-NEXT: 2.0
  // CHECK-NEXT: 3.0
  // CHECK-NEXT: 4.0
  // CHECK-NEXT: 5.0
  // CHECK-NEXT: 6.0
  // CHECK-NEXT: 7.0
  // CHECK-NEXT: 8.0
  // CHECK-NEXT: 9.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 11.0
  // CHECK-NEXT: data2:
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 20.0
  // CHECK-NEXT: 30.0
  // CHECK-NEXT: 40.0
  // CHECK-NEXT: 50.0
  // CHECK-NEXT: 60.0
  // CHECK-NEXT: 70.0
  // CHECK-NEXT: 80.0
  // CHECK-NEXT: 90.0
  // CHECK-NEXT: 100.0
  // CHECK-NEXT: 110.0

  // CHECK: device array values after update to:
  // CHECK-NEXT: data1:
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: data2:
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0

  free(data1);
  free(data2);
  return 0;
}
