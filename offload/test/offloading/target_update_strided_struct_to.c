// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
// This test checks that "update to" with struct member arrays supports strided
// sections using fixed-size arrays in structs.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16

typedef struct {
  double data[N];
  int len;
} T;

int main() {
  T s;
  s.len = N;

  // Initialize struct array on host with simple sequential values
  for (int i = 0; i < N; i++) {
    s.data[i] = i;
  }

  printf("original host struct array values:\n");
  for (int i = 0; i < N; i++)
    printf("%.1f\n", s.data[i]);
  printf("\n");

#pragma omp target data map(tofrom : s)
  {
// Initialize device struct array to 20
#pragma omp target map(tofrom : s)
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] = 20.0;
      }
    }

    // Modify host struct data for strided elements (set to 10)
    for (int i = 0; i < 8; i++) {
      s.data[i * 2] = 10.0; // Set even indices to 10
    }

// indices 0,2,4,6,8,10,12,14
#pragma omp target update to(s.data[0 : 8 : 2])

// Execute on device - add 5 to verify update worked
#pragma omp target map(tofrom : s)
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] += 5.0;
      }
    }
  }

  printf("after target update to struct:\n");
  for (int i = 0; i < N; i++)
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
  // CHECK-NEXT: 11.0
  // CHECK-NEXT: 12.0
  // CHECK-NEXT: 13.0
  // CHECK-NEXT: 14.0
  // CHECK-NEXT: 15.0

  // CHECK: after target update to struct:
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
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0
  // CHECK-NEXT: 15.0
  // CHECK-NEXT: 25.0

  return 0;
}
