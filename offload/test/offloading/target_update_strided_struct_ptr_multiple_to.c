// RUN: %libomptarget-compile-run-and-check-generic
// This test checks that multiple strided target updates to device work
// correctly with struct containing pointer-to-array member.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 12

typedef struct {
  double *data;
  int len;
} T;

#pragma omp declare mapper(custom : T v)                                       \
    map(tofrom : v, v.len, v.data[0 : v.len])

int main() {
  T s1, s2;
  s1.len = N;
  s1.data = (double *)calloc(N, sizeof(double));
  s2.len = N;
  s2.data = (double *)calloc(N, sizeof(double));

  // Initialize structs on host
  for (int i = 0; i < N; i++) {
    s1.data[i] = i;
    s2.data[i] = i;
  }

  printf("original s1 values:\n");
  for (int i = 0; i < N; i++)
    printf("%.1f ", s1.data[i]);
  printf("\n");

  printf("original s2 values:\n");
  for (int i = 0; i < N; i++)
    printf("%.1f ", s2.data[i]);
  printf("\n\n");

#pragma omp target data map(tofrom : s1, s2)
  {
// Initialize device struct arrays to 20
#pragma omp target
    {
      for (int i = 0; i < s1.len; i++) {
        s1.data[i] = 20.0;
        s2.data[i] = 20.0;
      }
    }

    // Modify host: s1 even indices, s2 multiples of 3
    for (int i = 0; i < s1.len; i += 2) {
      s1.data[i] = 10.0;
    }
    for (int i = 0; i < s2.len; i += 3) {
      s2.data[i] = 10.0;
    }

// Multiple strided updates to device: s1 even (s1.len/2 elements, stride 2), s2
// multiples of 3 (s2.len/3 elements, stride 3)
#pragma omp target update to(s1.data[0 : s1.len / 2 : 2],                      \
                                 s2.data[0 : s2.len / 3 : 3])

// Verify update on device by adding 5
#pragma omp target
    {
      for (int i = 0; i < s1.len; i++) {
        s1.data[i] += 5.0;
        s2.data[i] += 5.0;
      }
    }
  } // Exit target data - tofrom mapper copies data back

  printf("s1 after update to device (even indices):\n");
  for (int i = 0; i < N; i++)
    printf("%.1f ", s1.data[i]);
  printf("\n");

  printf("s2 after update to device (multiples of 3):\n");
  for (int i = 0; i < N; i++)
    printf("%.1f ", s2.data[i]);
  printf("\n");

  // CHECK: original s1 values:
  // CHECK-NEXT: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0

  // CHECK: original s2 values:
  // CHECK-NEXT: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0

  // CHECK: s1 after update to device (even indices):
  // CHECK-NEXT: 15.0 25.0 15.0 25.0 15.0 25.0 15.0 25.0 15.0 25.0 15.0 25.0

  // CHECK: s2 after update to device (multiples of 3):
  // CHECK-NEXT: 15.0 25.0 25.0 15.0 25.0 25.0 15.0 25.0 25.0 15.0 25.0 25.0

  free(s1.data);
  free(s2.data);
  return 0;
}
