// RUN: %libomptarget-compile-run-and-check-generic
// This test checks that #pragma omp target update to(s.data[0:N/5:5]) correctly
// updates partial strided elements (stride larger than update count) from host
// to device using a struct with pointer-to-array member.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 20

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
    printf("%.1f ", s.data[i]);
  printf("\n\n");

#pragma omp target data map(tofrom : s)
  {
// Initialize device struct arrays to 20
#pragma omp target
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] = 20.0;
      }
    }

    // Modify host elements: indices 0, 5, 10, 15 only
    s.data[0] = 10.0;
    s.data[5] = 10.0;
    s.data[10] = 10.0;
    s.data[15] = 10.0;

// Update indices 0, 5, 10, 15 only (N/5 = 4 elements with stride 5) to device
#pragma omp target update to(s.data[0 : N / 5 : 5])

// Execute on device - add 5 to verify update worked
#pragma omp target
    {
      for (int i = 0; i < s.len; i++) {
        s.data[i] += 5.0;
      }
    }
  } // Exit target data - tofrom mapper copies data back

  printf("device array values after partial stride update to:\n");
  for (int i = 0; i < N; i++)
    printf("%.1f ", s.data[i]);
  printf("\n");

  // CHECK: original host array values:
  // CHECK-NEXT:
  // 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0
  // 18.0 19.0

  // CHECK: device array values after partial stride update to:
  // CHECK-NEXT: 15.0 25.0 25.0 25.0 25.0 15.0 25.0 25.0 25.0 25.0 15.0 25.0 25.0
  // 25.0 25.0 15.0 25.0 25.0 25.0 25.0

  free(s.data);
  return 0;
}
