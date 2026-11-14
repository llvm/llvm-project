// RUN: %libomptarget-compile-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
// UNSUPPORTED: apu

// A 2-D strided target update from of an array of structs whose pointers are
// all attached. arr[0:2][1:2:2] covers only arr[i][1] and arr[i][3]:
//
//   arr[i]:  [ p0 x0 ][ p1 x1 ][ p2 x2 ][ p3 x3 ]
//   update:           [=======]         [=======]
//
// The host pointers p1/p3 keep their values from before the transfer, while
// p0/p2 and x0/x2 are not touched, even though all pointers were re-pointed
// on the host (allowed since OpenMP 6.1).

#include <stdio.h>

typedef struct {
  int *p;
  int x;
} S;

int a[8], b[8];
S arr[2][4];

int main() {
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      arr[i][j] = (S){&a[4 * i + j], 0};
#pragma omp target enter data map(to : a)
#pragma omp target enter data map(                                             \
        to : arr, arr[0][0].p[0 : 1], arr[0][1].p[0 : 1], arr[0][2].p[0 : 1],  \
            arr[0][3].p[0 : 1], arr[1][0].p[0 : 1], arr[1][1].p[0 : 1],        \
            arr[1][2].p[0 : 1], arr[1][3].p[0 : 1])

#pragma omp target
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      arr[i][j].x = 10 * i + j + 1;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      arr[i][j].p = &b[4 * i + j];
#pragma omp target update from(arr[0 : 2][1 : 2 : 2])

  int ok = 1;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      ok &= arr[i][j].p == &b[4 * i + j] &&
            arr[i][j].x == (j % 2 ? 10 * i + j + 1 : 0);
  // CHECK: update from: ok
  printf("update from: %s\n", ok ? "ok" : "wrong");

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      arr[i][j].p = &a[4 * i + j];
#pragma omp target exit data map(delete : arr, a)
  return 0;
}
