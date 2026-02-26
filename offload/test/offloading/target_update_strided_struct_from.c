// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
// This test checks that "update from" with user-defined mapper supports strided
// sections using fixed-size arrays in structs.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16

typedef struct {
  double data[N];
  size_t len;
} T;

#pragma omp declare mapper(custom : T v) map(to : v, v.len, v.data[0 : v.len])

int main() {
  T s;
  s.len = N;

  for (int i = 0; i < N; i++) {
    s.data[i] = i;
  }

  printf("original host array values:\n");
  for (int i = 0; i < N; i++)
    printf("%f\n", s.data[i]);
  printf("\n");

#pragma omp target data map(mapper(custom), to : s)
  {
// Execute on device with explicit mapper
#pragma omp target map(mapper(custom), tofrom : s)
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] += i;
      }
    }

// Update strided elements from device: indices 0,2,4,6,8,10,12,14
#pragma omp target update from(s.data[0 : 8 : 2])
  }

  printf("from target array results:\n");
  for (int i = 0; i < N; i++)
    printf("%f\n", s.data[i]);

  // CHECK: original host array values:
  // CHECK-NEXT: 0.000000
  // CHECK-NEXT: 1.000000
  // CHECK-NEXT: 2.000000
  // CHECK-NEXT: 3.000000
  // CHECK-NEXT: 4.000000
  // CHECK-NEXT: 5.000000
  // CHECK-NEXT: 6.000000
  // CHECK-NEXT: 7.000000
  // CHECK-NEXT: 8.000000
  // CHECK-NEXT: 9.000000
  // CHECK-NEXT: 10.000000
  // CHECK-NEXT: 11.000000
  // CHECK-NEXT: 12.000000
  // CHECK-NEXT: 13.000000
  // CHECK-NEXT: 14.000000
  // CHECK-NEXT: 15.000000

  // CHECK: from target array results:
  // CHECK-NEXT: 0.000000
  // CHECK-NEXT: 1.000000
  // CHECK-NEXT: 4.000000
  // CHECK-NEXT: 3.000000
  // CHECK-NEXT: 8.000000
  // CHECK-NEXT: 5.000000
  // CHECK-NEXT: 12.000000
  // CHECK-NEXT: 7.000000
  // CHECK-NEXT: 16.000000
  // CHECK-NEXT: 9.000000
  // CHECK-NEXT: 20.000000
  // CHECK-NEXT: 11.000000
  // CHECK-NEXT: 24.000000
  // CHECK-NEXT: 13.000000
  // CHECK-NEXT: 28.000000
  // CHECK-NEXT: 15.000000

  return 0;
}
