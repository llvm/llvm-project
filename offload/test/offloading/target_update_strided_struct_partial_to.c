// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
// This test checks that #pragma omp target update to(s.data[0:2:3]) correctly
// updates every third element (stride 3) from the host to the device
// for struct member arrays.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define LEN 11

typedef struct {
  double data[LEN];
  int len;
} T;

int main() {
  T s;
  s.len = LEN;

  // Initialize struct array on host with simple sequential values
  for (int i = 0; i < LEN; i++)
    s.data[i] = i;

  printf("original host struct array values:\n");
  for (int i = 0; i < LEN; i++)
    printf("%.1f\n", s.data[i]);
  printf("\n");

#pragma omp target data map(tofrom : s)
  {
// Initialize all elements on device to 20
#pragma omp target map(tofrom : s)
    {
      for (int i = 0; i < s.len; i++)
        s.data[i] = 20.0;
    }

    // Modify host struct data for elements that will be updated (set to 10)
    s.data[0] = 10.0;
    s.data[3] = 10.0;

// indices 0,3 only
#pragma omp target update to(s.data[0 : 2 : 3])

// Verify on device by adding 5 to all elements
#pragma omp target map(tofrom : s)
    {
      for (int i = 0; i < s.len; i++)
        s.data[i] += 5.0;
    }
  }

  printf("device struct array values after update to:\n");
  for (int i = 0; i < LEN; i++)
    printf("%.1f\n", s.data[i]);

  // CHECK: original host struct array values:
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

  // CHECK: device struct array values after update to:
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 25.0

  return 0;
}
