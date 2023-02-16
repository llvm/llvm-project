// RUN: %libomp-compile-and-run

#include <stdio.h>

#define N 3

int arr[N][N][N];
int main() {
#pragma omp for collapse(3)
  for (unsigned int i = 0; i < N; ++i)
    for (unsigned int j = i; j < N; ++j)
      for (unsigned int k = j; k < N; ++k)
        arr[i][j][k] = 1;
  int num_failed = 0;
  for (unsigned int i = 0; i < N; ++i)
    for (unsigned int j = 0; j < N; ++j)
      for (unsigned int k = 0; k < N; ++k)
        if (arr[i][j][k] == (j >= i && k >= j) ? 0 : 1)
          ++num_failed;

  return num_failed;
}
