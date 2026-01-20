// RUN: %libomptarget-compile-run-and-check-generic
// This test checks that #pragma omp target update to(s.data[0:s.len/2:2])
// correctly updates every second element (stride 2) from the host to the device
// using a struct with pointer-to-array member.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16

typedef struct {
  double *data;
  int len;
} T;

#pragma omp declare mapper(custom : T v)                                       \
    map(tofrom : v, v.len, v.data[0 : v.len])

int main() {
  T s;
  s.len = N;
  s.data = (double *)calloc(N, sizeof(double));

  // Initialize struct data on host
  for (int i = 0; i < N; i++) {
    s.data[i] = i;
  }

  printf("original host array values:\n");
  for (int i = 0; i < N; i++)
    printf("%.1f\n", s.data[i]);
  printf("\n");

#pragma omp target data map(tofrom : s)
  {
// Set device data to 20
#pragma omp target
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] = 20.0;
      }
    }

    // Modify host even-indexed elements
    for (int i = 0; i < N; i += 2) {
      s.data[i] = 10.0;
    }

// Update only even indices (0,2,4,6,8,10,12,14) to device - s.len/2 elements
// with stride 2
#pragma omp target update to(s.data[0 : s.len / 2 : 2])

// Execute on device - add 5 to verify update worked
#pragma omp target
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] += 5.0;
      }
    }
  } // Exit target data - tofrom mapper copies data back

  printf("device array values after update to:\n");
  for (int i = 0; i < N; i++)
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
  // CHECK-NEXT: 11.0
  // CHECK-NEXT: 12.0
  // CHECK-NEXT: 13.0
  // CHECK-NEXT: 14.0
  // CHECK-NEXT: 15.0

  // CHECK: device array values after update to:
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

  free(s.data);
  return 0;
}
