// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu
// Make sure multi-dimensional strided offset update works correctly.

#include <stdio.h>

#define N 6
#define SLICE1 1 : 2 : 2
#define SLICE2 2 : 3
int main() {
  int darr[N][N];

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      darr[i][j] = 100 + 10 * i + j;

  printf("Full array\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("\t%d", darr[i][j]);
    }
    printf("\n");
  }

#pragma omp target enter data map(alloc : darr[0 : N][0 : N])

  // Zero out target array.
#pragma omp target
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      darr[i][j] = 0;

  // Only copy over the slice to the device.
#pragma omp target update to(darr[SLICE1][SLICE2])
  // Then copy over the entire array to the host.
#pragma omp target exit data map(from : darr[0 : N][0 : N])

  printf("Only slice (to)\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("\t%d", darr[i][j]);
    }
    printf("\n");
  }

#pragma omp target enter data map(alloc : darr[0 : N][0 : N])

  // Initialize on the device
#pragma omp target
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      darr[i][j] = 100 + 10 * i + j;

  // Zero out host array.
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      darr[i][j] = 0;

  // Copy over only the slice to the host
#pragma omp target update from(darr[SLICE1][SLICE2])
#pragma omp target exit data map(delete : darr[0 : N][0 : N])

  printf("Only slice (from)\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("\t%d", darr[i][j]);
    }
    printf("\n");
  }

  return 0;
}

//      CHECK: Full array
// CHECK-NEXT:         100     101     102     103     104     105
// CHECK-NEXT:         110     111     112     113     114     115
// CHECK-NEXT:         120     121     122     123     124     125
// CHECK-NEXT:         130     131     132     133     134     135
// CHECK-NEXT:         140     141     142     143     144     145
// CHECK-NEXT:         150     151     152     153     154     155
// CHECK-NEXT: Only slice (to)
// CHECK-NEXT:         0       0       0       0       0       0
// CHECK-NEXT:         0       0       112     113     114     0
// CHECK-NEXT:         0       0       0       0       0       0
// CHECK-NEXT:         0       0       132     133     134     0
// CHECK-NEXT:         0       0       0       0       0       0
// CHECK-NEXT:         0       0       0       0       0       0
// CHECK-NEXT: Only slice (from)
// CHECK-NEXT:         0       0       0       0       0       0
// CHECK-NEXT:         0       0       112     113     114     0
// CHECK-NEXT:         0       0       0       0       0       0
// CHECK-NEXT:         0       0       132     133     134     0
// CHECK-NEXT:         0       0       0       0       0       0
// CHECK-NEXT:         0       0       0       0       0       0
