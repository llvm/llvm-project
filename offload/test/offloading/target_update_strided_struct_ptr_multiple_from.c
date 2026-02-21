// RUN: %libomptarget-compile-run-and-check-generic
// This test checks that multiple strided target updates work correctly
// with struct containing pointer-to-array member.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 12

typedef struct {
  double *data;
  int len;
} T;

#pragma omp declare mapper(custom : T v) map(to : v, v.len, v.data[0 : v.len])

int main() {
  T s1, s2;
  s1.len = N;
  s1.data = (double *)calloc(N, sizeof(double));
  s2.len = N;
  s2.data = (double *)calloc(N, sizeof(double));

  printf("original s1 values:\n");
  for (int i = 0; i < N; i++)
    printf("%.1f ", s1.data[i]);
  printf("\n");

  printf("original s2 values:\n");
  for (int i = 0; i < N; i++)
    printf("%.1f ", s2.data[i]);
  printf("\n\n");

#pragma omp target data map(mapper(custom), to : s1, s2)
  {
// Modify on device
#pragma omp target
    {
      // s1: set even indices to 10
      for (int i = 0; i < s1.len; i += 2) {
        s1.data[i] = 10.0;
      }
      // s2: set multiples of 3 to 10
      for (int i = 0; i < s2.len; i += 3) {
        s2.data[i] = 10.0;
      }
    }

// Multiple strided updates: s1 even (s1.len/2 elements, stride 2), s2 multiples
// of 3 (s2.len/3 elements, stride 3)
#pragma omp target update from(s1.data[0 : s1.len / 2 : 2],                    \
                                   s2.data[0 : s2.len / 3 : 3])
  }

  printf("s1 after update (even indices):\n");
  for (int i = 0; i < N; i++)
    printf("%.1f ", s1.data[i]);
  printf("\n");

  printf("s2 after update (multiples of 3):\n");
  for (int i = 0; i < N; i++)
    printf("%.1f ", s2.data[i]);
  printf("\n");

  // CHECK: original s1 values:
  // CHECK-NEXT: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

  // CHECK: original s2 values:
  // CHECK-NEXT: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

  // CHECK: s1 after update (even indices):
  // CHECK-NEXT: 10.0 0.0 10.0 0.0 10.0 0.0 10.0 0.0 10.0 0.0 10.0 0.0

  // CHECK: s2 after update (multiples of 3):
  // CHECK-NEXT: 10.0 0.0 0.0 10.0 0.0 0.0 10.0 0.0 0.0 10.0 0.0 0.0

  free(s1.data);
  free(s2.data);
  return 0;
}
