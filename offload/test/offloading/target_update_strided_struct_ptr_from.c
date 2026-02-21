// RUN: %libomptarget-compile-run-and-check-generic
// This test checks that #pragma omp target update from(s.data[0:s.len/2:2])
// correctly updates every second element (stride 2) from the device to the host
// using a struct with pointer-to-array member.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16

typedef struct {
  double *data;
  int len;
} T;

#pragma omp declare mapper(custom : T v) map(to : v, v.len, v.data[0 : v.len])

int main() {
  T s;
  s.len = N;
  s.data = (double *)calloc(N, sizeof(double));

  printf("original host array values:\n");
  for (int i = 0; i < N; i++)
    printf("%.1f\n", s.data[i]);
  printf("\n");

#pragma omp target data map(mapper(custom), to : s)
  {
// Execute on device - modify even-indexed elements
#pragma omp target
    {
      for (int i = 0; i < s.len; i += 2) {
        s.data[i] = 10.0;
      }
    }

// Update only even indices (0,2,4,6,8,10,12,14) - s.len/2 elements with stride
// 2
#pragma omp target update from(s.data[0 : s.len / 2 : 2])
  }

  printf("device array values after update from:\n");
  for (int i = 0; i < N; i++)
    printf("%.1f\n", s.data[i]);
  printf("\n");

  // CHECK: original host array values:
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 0.0

  // CHECK: device array values after update from:
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 0.0
  // CHECK-NEXT: 10.0
  // CHECK-NEXT: 0.0

  free(s.data);
  return 0;
}
