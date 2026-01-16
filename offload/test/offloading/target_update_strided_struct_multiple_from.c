// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
// This test checks that #pragma omp target update from(s1.data[0:6:2],
// s2.data[0:4:3]) correctly updates strided sections covering the full arrays
// from device to host.

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

#pragma omp target data map(to : s1, s2)
  {
// Initialize all device values to 20
#pragma omp target map(tofrom : s1, s2)
    {
      for (int i = 0; i < s1.len; i++) {
        s1.data[i] = 20.0;
        s2.data[i] = 20.0;
      }
    }

// Modify specific strided elements on device to 10
#pragma omp target map(tofrom : s1, s2)
    {
      // s1: modify even indices (0,2,4,6,8,10)
      for (int i = 0; i < 6; i++) {
        s1.data[i * 2] = 10.0;
      }
      // s2: modify every 3rd index (0,3,6,9)
      for (int i = 0; i < 4; i++) {
        s2.data[i * 3] = 10.0;
      }
    }

// s1.data[0:6:2] updates only even indices: 0,2,4,6,8,10
// s2.data[0:4:3] updates only every 3rd: 0,3,6,9
#pragma omp target update from(s1.data[0 : 6 : 2], s2.data[0 : 4 : 3])
  }

  printf("host struct array values after update from:\n");
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

  // CHECK: host struct array values after update from:
  // CHECK-NEXT: s1.data:
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 1.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 3.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 5.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 7.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 9.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 11.0
  // CHECK-NEXT: s2.data:
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 1.0
  // CHECK-NEXT: 2.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 4.0
  // CHECK-NEXT: 5.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 7.0
  // CHECK-NEXT: 8.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 11.0

  return 0;
}
