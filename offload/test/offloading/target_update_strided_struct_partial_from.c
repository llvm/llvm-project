// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
// This test checks that #pragma omp target update from(s.data[0:2:3]) correctly
// updates every third element (stride 3) from the device to the host
// using a struct with fixed-size array member.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define LEN 11

typedef struct {
  double data[LEN];
  size_t len;
} T;

#pragma omp declare mapper(custom : T v) map(to : v, v.len, v.data[0 : v.len])

int main() {
  T s;
  s.len = LEN;

  // Initialize struct data on host
  for (int i = 0; i < LEN; i++) {
    s.data[i] = i;
  }

  printf("original host array values:\n");
  for (int i = 0; i < LEN; i++)
    printf("%.1f\n", s.data[i]);
  printf("\n");

#pragma omp target data map(mapper(custom), to : s)
  {
// Execute on device with mapper
#pragma omp target map(mapper(custom), tofrom : s)
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] = 20.0; // Set all to 20 on device
      }
    }

// Modify specific elements on device (only first 2 stride positions)
#pragma omp target map(mapper(custom), tofrom : s)
    {
      s.data[0] = 10.0;
      s.data[3] = 10.0;
    }

// indices 0,3 only
#pragma omp target update from(s.data[0 : 2 : 3])
  }

  printf("device array values after update from:\n");
  for (int i = 0; i < LEN; i++)
    printf("%.1f\n", s.data[i]);
  printf("\n");

  // CHECK: original host array values:
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

  // CHECK: device array values after update from:
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 1.0
  // CHECK-NEXT: 2.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 4.0
  // CHECK-NEXT: 5.0
  // CHECK-NEXT: 6.0
  // CHECK-NEXT: 7.0
  // CHECK-NEXT: 8.0
  // CHECK-NEXT: 9.0
  // CHECK-NEXT: 10.0

  return 0;
}
