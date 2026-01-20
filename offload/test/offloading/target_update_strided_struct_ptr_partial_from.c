// RUN: %libomptarget-compile-run-and-check-generic
// This test checks that #pragma omp target update from(s.data[0:N/5:5])
// correctly updates partial strided elements (stride larger than update count)
// from device to host using a struct with pointer-to-array member.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 20

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
    printf("%.1f ", s.data[i]);
  printf("\n\n");

#pragma omp target data map(mapper(custom), tofrom : s)
  {
// Set all elements to 20 on device
#pragma omp target map(mapper(custom), tofrom : s)
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] = 20.0; // Set all to 20 on device
      }
    }

// Modify specific elements on device (only first 4 stride positions)
#pragma omp target map(mapper(custom), tofrom : s)
    {
      s.data[0] = 10.0;
      s.data[5] = 10.0;
      s.data[10] = 10.0;
      s.data[15] = 10.0;
    }

// Update indices 0, 5, 10, 15 only (N/5 = 4 elements with stride 5)
#pragma omp target update from(s.data[0 : N / 5 : 5])
  }

  printf("device array values after partial stride update:\n");
  for (int i = 0; i < N; i++)
    printf("%.1f ", s.data[i]);
  printf("\n");

  // CHECK: original host array values:
  // CHECK-NEXT: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  // 0.0 0.0 0.0 0.0

  // CHECK: device array values after partial stride update:
  // CHECK-NEXT: 10.0 0.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0
  // 0.0 10.0 0.0 0.0 0.0 0.0

  free(s.data);
  return 0;
}
