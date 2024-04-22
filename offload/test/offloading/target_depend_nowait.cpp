// RUN: %libomptarget-compilexx-run-and-check-generic

#include <omp.h>
#include <stdio.h>

#define N 1024

int A[N];
int B[N];
int C[N];
int main() {
  for (int i = 0; i < N; i++)
    A[i] = B[i] = i;

#pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 1) {
// map data A & B and move to
#pragma omp target enter data map(to : A, B) depend(out : A[0]) nowait

// no data move since already mapped
#pragma omp target map(A, B) depend(out : A[0]) nowait
      {
        for (int i = 0; i < N; i++)
          ++A[i];
        for (int i = 0; i < N; i++)
          ++B[i];
      }

// no data move since already mapped
#pragma omp target teams num_teams(1) map(A, B) depend(out : A[0]) nowait
      {
        for (int i = 0; i < N; i++)
          ++A[i];
        for (int i = 0; i < N; i++)
          ++B[i];
      }

// A updated via update
#pragma omp target update from(A) depend(out : A[0]) nowait

// B updated via exit, A just released
#pragma omp target exit data map(release : A) map(from : B) depend(out : A[0]) \
    nowait
    } // if
  }   // parallel

  int Sum = 0;
  for (int i = 0; i < N; i++)
    Sum += A[i] + B[i];
  // Sum is 2 * N * (2 + N - 1 + 2) / 2
  // CHECK: Sum = 1051648.
  printf("Sum = %d.\n", Sum);

  return Sum != 2 * N * (2 + N - 1 + 2) / 2;
}
