// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
// This test checks that #pragma omp target update to(s1.data[0:6:2],
// s2.data[0:4:3]) correctly updates strided sections covering the full arrays
// from host to device.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define LEN 12

typedef struct {
  double data[LEN];
  int len;
} T;

int main() {
  T s1, s2;
  s1.len = LEN;
  s2.len = LEN;

  // Initialize struct arrays on host with simple sequential values
  for (int i = 0; i < LEN; i++) {
    s1.data[i] = i;
    s2.data[i] = i;
  }

  printf("original host struct array values:\n");
  printf("s1.data:\n");
  for (int i = 0; i < LEN; i++)
    printf("%.1f\n", s1.data[i]);
  printf("s2.data:\n");
  for (int i = 0; i < LEN; i++)
    printf("%.1f\n", s2.data[i]);
  printf("\n");

#pragma omp target data map(tofrom : s1, s2)
  {
// Initialize device struct arrays to 20
#pragma omp target map(tofrom : s1, s2)
    {
      for (int i = 0; i < s1.len; i++) {
        s1.data[i] = 20.0;
        s2.data[i] = 20.0;
      }
    }

    // s1: even indices (0,2,4,6,8,10)
    for (int i = 0; i < 6; i++) {
      s1.data[i * 2] = 10.0;
    }
    // s2: every 3rd index (0,3,6,9)
    for (int i = 0; i < 4; i++) {
      s2.data[i * 3] = 10.0;
    }

// s1.data[0:6:2] updates all even indices: 0,2,4,6,8,10 (6 elements, stride 2)
// s2.data[0:4:3] updates every 3rd: 0,3,6,9 (4 elements, stride 3)
#pragma omp target update to(s1.data[0 : 6 : 2], s2.data[0 : 4 : 3])

// Verify update on device by adding 5
#pragma omp target map(tofrom : s1, s2)
    {
      for (int i = 0; i < s1.len; i++) {
        s1.data[i] += 5.0;
        s2.data[i] += 5.0;
      }
    }
  }

  printf("device struct array values after update to:\n");
  printf("s1.data:\n");
  for (int i = 0; i < LEN; i++)
    printf("%.1f\n", s1.data[i]);
  printf("s2.data:\n");
  for (int i = 0; i < LEN; i++)
    printf("%.1f\n", s2.data[i]);

  // CHECK: original host struct array values:
  // CHECK-NEXT: s1.data:
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
  // CHECK-NEXT: s2.data:
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

  // CHECK: device struct array values after update to:
  // CHECK-NEXT: s1.data:
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
  // CHECK-NEXT: s2.data:
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

  return 0;
}
