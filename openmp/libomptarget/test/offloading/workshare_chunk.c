// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

// clang-format off

#include <omp.h>
#include <stdio.h>

#define N 100
#define BLOCK_SHIFT 8

void print(int *A, int size) {
  for (int i = 0; i < size; ++i) {
    printf("B%dT%d ", A[i] >> BLOCK_SHIFT, A[i] % (1 << BLOCK_SHIFT));
  }
  printf("\n");
}

int main() {
  int A[N];

#pragma omp target parallel for map(from:A) num_threads(10) schedule(static, 2)
  for (int i = 0; i < N; ++i) {
     A[i] = (omp_get_team_num() << BLOCK_SHIFT) + omp_get_thread_num();
  }
  printf("omp target parallel for thread chunk size %d\n", 2);
  print(A, N);

#pragma omp target teams distribute map(from:A) num_teams(10) \
        dist_schedule(static, 2)
  for (int i = 0; i < N; ++i) {
     A[i] = (omp_get_team_num() << BLOCK_SHIFT) + omp_get_thread_num();
  }
  printf("omp target teams distribute block chunk size %d\n", 2);
  print(A, N);

#pragma omp target teams distribute parallel for map(from:A) \
        num_teams(10) dist_schedule(static, 2)
  for (int i = 0; i < N; ++i) {
     A[i] = (omp_get_team_num() << BLOCK_SHIFT) + omp_get_thread_num();
  }
  printf("omp target teams distribute parallel for block chunk size %d ", 2);
  printf("thread chunk size default\n");
  print(A, N);

#pragma omp target teams distribute parallel for map(from:A) \
        num_teams(10) dist_schedule(static, 2) schedule(static, 3)
  for (int i = 0; i < N; ++i) {
     A[i] = (omp_get_team_num() << BLOCK_SHIFT) + omp_get_thread_num();
  }
  printf("omp target teams distribute parallel for block chunk size %d ", 2);
  printf("thread chunk size %d\n", 3);
  print(A, N);

#pragma omp target teams distribute parallel for map(from:A) \
        num_teams(10) dist_schedule(static, 3) schedule(static, 2)
  for (int i = 0; i < N; ++i) {
     A[i] = (omp_get_team_num() << BLOCK_SHIFT) + omp_get_thread_num();
  }
  printf("omp target teams distribute parallel for block chunk size %d ", 3);
  printf("thread chunk size %d\n", 2);
  print(A, N);

#pragma omp target teams distribute parallel for map(from:A) \
        num_teams(10) dist_schedule(static, 5) schedule(static, 2)
  for (int i = 0; i < N; ++i) {
     A[i] = (omp_get_team_num() << BLOCK_SHIFT) + omp_get_thread_num();
  }
  printf("omp target teams distribute parallel for block chunk size %d ", 5);
  printf("thread chunk size %d\n", 2);
  print(A, N);

#pragma omp target teams distribute parallel for map(from:A) num_teams(10) \
        dist_schedule(static, 49) schedule(static, 2)
  for (int i = 0; i < N; ++i) {
     A[i] = (omp_get_team_num() << BLOCK_SHIFT) + omp_get_thread_num();
  }
  printf("omp target teams distribute parallel for block chunk size %d ", 49);
  printf("thread chunk size %d\n", 2);
  print(A, N);

#pragma omp target teams distribute parallel for map(from:A) \
        num_teams(10) num_threads(10) dist_schedule(static, 29)
  for (int i = 0; i < N; ++i) {
     A[i] = (omp_get_team_num() << BLOCK_SHIFT) + omp_get_thread_num();
  }
  printf("omp target teams distribute parallel for block chunk size %d ", 29);
  printf("thread chunk size default\n");
  print(A, N);

#pragma omp target teams distribute parallel for map(from:A) \
        num_teams(10) num_threads(10) dist_schedule(static, 101)
  for (int i = 0; i < N; ++i) {
     A[i] = (omp_get_team_num() << BLOCK_SHIFT) + omp_get_thread_num();
  }
  printf("omp target teams distribute parallel for block chunk size %d ", 101);
  printf("thread chunk size default\n");
  print(A, N);

#pragma omp target teams distribute parallel for map(from:A) \
        num_teams(9) num_threads(10) schedule(static, 101)
  for (int i = 0; i < N; ++i) {
     A[i] = (omp_get_team_num() << BLOCK_SHIFT) + omp_get_thread_num();
  }
  printf("omp target teams distribute parallel for default block chunk size ");
  printf("thread chunk size %d\n", 101);
  print(A, N);
  return 0;
}
//CHECK:      omp target parallel for thread chunk size 2

//CHECK-NEXT: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4
//CHECK-SAME: B0T5 B0T5 B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9
//CHECK-SAME: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4
//CHECK-SAME: B0T5 B0T5 B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9
//CHECK-SAME: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4
//CHECK-SAME: B0T5 B0T5 B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9
//CHECK-SAME: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4
//CHECK-SAME: B0T5 B0T5 B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9
//CHECK-SAME: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4
//CHECK-SAME: B0T5 B0T5 B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9

//CHECK:      omp target teams distribute block chunk size 2

//CHECK-NEXT: B0T0 B0T0 B1T0 B1T0 B2T0 B2T0 B3T0 B3T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B6T0 B6T0 B7T0 B7T0 B8T0 B8T0 B9T0 B9T0
//CHECK-SAME: B0T0 B0T0 B1T0 B1T0 B2T0 B2T0 B3T0 B3T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B6T0 B6T0 B7T0 B7T0 B8T0 B8T0 B9T0 B9T0
//CHECK-SAME: B0T0 B0T0 B1T0 B1T0 B2T0 B2T0 B3T0 B3T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B6T0 B6T0 B7T0 B7T0 B8T0 B8T0 B9T0 B9T0
//CHECK-SAME: B0T0 B0T0 B1T0 B1T0 B2T0 B2T0 B3T0 B3T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B6T0 B6T0 B7T0 B7T0 B8T0 B8T0 B9T0 B9T0
//CHECK-SAME: B0T0 B0T0 B1T0 B1T0 B2T0 B2T0 B3T0 B3T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B6T0 B6T0 B7T0 B7T0 B8T0 B8T0 B9T0 B9T0

//CHECK:      omp target teams distribute parallel for
//CHECK-SAME: block chunk size 2 thread chunk size default

//CHECK-NEXT: B0T0 B0T1 B1T0 B1T1 B2T0 B2T1 B3T0 B3T1 B4T0 B4T1
//CHECK-SAME: B5T0 B5T1 B6T0 B6T1 B7T0 B7T1 B8T0 B8T1 B9T0 B9T1
//CHECK-SAME: B0T0 B0T1 B1T0 B1T1 B2T0 B2T1 B3T0 B3T1 B4T0 B4T1
//CHECK-SAME: B5T0 B5T1 B6T0 B6T1 B7T0 B7T1 B8T0 B8T1 B9T0 B9T1
//CHECK-SAME: B0T0 B0T1 B1T0 B1T1 B2T0 B2T1 B3T0 B3T1 B4T0 B4T1
//CHECK-SAME: B5T0 B5T1 B6T0 B6T1 B7T0 B7T1 B8T0 B8T1 B9T0 B9T1

//CHECK:      omp target teams distribute parallel for
//CHECK-SAME  block chunk size 2 thread chunk size 3

//CHECK-NEXT: B0T0 B0T0 B1T0 B1T0 B2T0 B2T0 B3T0 B3T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B6T0 B6T0 B7T0 B7T0 B8T0 B8T0 B9T0 B9T0
//CHECK-SAME: B0T0 B0T0 B1T0 B1T0 B2T0 B2T0 B3T0 B3T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B6T0 B6T0 B7T0 B7T0 B8T0 B8T0 B9T0 B9T0
//CHECK-SAME: B0T0 B0T0 B1T0 B1T0 B2T0 B2T0 B3T0 B3T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B6T0 B6T0 B7T0 B7T0 B8T0 B8T0 B9T0 B9T0
//CHECK-SAME: B0T0 B0T0 B1T0 B1T0 B2T0 B2T0 B3T0 B3T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B6T0 B6T0 B7T0 B7T0 B8T0 B8T0 B9T0 B9T0
//CHECK-SAME: B0T0 B0T0 B1T0 B1T0 B2T0 B2T0 B3T0 B3T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B6T0 B6T0 B7T0 B7T0 B8T0 B8T0 B9T0 B9T0

//CHECK:      omp target teams distribute parallel for
//CHECK-SAME: block chunk size 3 thread chunk size 2

//CHECK-NEXT: B0T0 B0T0 B0T1 B1T0 B1T0 B1T1 B2T0 B2T0 B2T1
//CHECK-SAME: B3T0 B3T0 B3T1 B4T0 B4T0 B4T1
//CHECK-SAME: B5T0 B5T0 B5T1 B6T0 B6T0 B6T1 B7T0 B7T0 B7T1
//CHECK-SAME: B8T0 B8T0 B8T1 B9T0 B9T0 B9T1
//CHECK-SAME: B0T0 B0T0 B0T1 B1T0 B1T0 B1T1 B2T0 B2T0 B2T1
//CHECK-SAME: B3T0 B3T0 B3T1 B4T0 B4T0 B4T1
//CHECK-SAME: B5T0 B5T0 B5T1 B6T0 B6T0 B6T1 B7T0 B7T0 B7T1
//CHECK-SAME: B8T0 B8T0 B8T1 B9T0 B9T0 B9T1
//CHECK-SAME: B0T0 B0T0 B0T1 B1T0 B1T0 B1T1 B2T0 B2T0 B2T1
//CHECK-SAME: B3T0 B3T0 B3T1 B4T0 B4T0 B4T1
//CHECK-SAME: B5T0 B5T0 B5T1 B6T0 B6T0 B6T1 B7T0 B7T0 B7T1
//CHECK-SAME: B8T0 B8T0 B8T1 B9T0 B9T0 B9T1
//CHECK-SAME: B0T0 B0T0 B0T1 B1T0 B1T0 B1T1 B2T0 B2T0 B2T1 B3T0

//CHECK:      omp target teams distribute parallel for
//CHECK-SAME: block chunk size 5 thread chunk size 2

//CHECK-NEXT: B0T0 B0T0 B0T1 B0T1 B0T2 B1T0 B1T0 B1T1 B1T1 B1T2
//CHECK-SAME: B2T0 B2T0 B2T1 B2T1 B2T2 B3T0 B3T0 B3T1 B3T1 B3T2
//CHECK-SAME: B4T0 B4T0 B4T1 B4T1 B4T2 B5T0 B5T0 B5T1 B5T1 B5T2
//CHECK-SAME: B6T0 B6T0 B6T1 B6T1 B6T2 B7T0 B7T0 B7T1 B7T1 B7T2
//CHECK-SAME: B8T0 B8T0 B8T1 B8T1 B8T2 B9T0 B9T0 B9T1 B9T1 B9T2
//CHECK-SAME: B0T0 B0T0 B0T1 B0T1 B0T2 B1T0 B1T0 B1T1 B1T1 B1T2
//CHECK-SAME: B2T0 B2T0 B2T1 B2T1 B2T2 B3T0 B3T0 B3T1 B3T1 B3T2
//CHECK-SAME: B4T0 B4T0 B4T1 B4T1 B4T2 B5T0 B5T0 B5T1 B5T1 B5T2
//CHECK-SAME: B6T0 B6T0 B6T1 B6T1 B6T2 B7T0 B7T0 B7T1 B7T1 B7T2
//CHECK-SAME: B8T0 B8T0 B8T1 B8T1 B8T2 B9T0 B9T0 B9T1 B9T1 B9T2

//CHECK:      omp target teams distribute parallel for
//CHECK-SAME: block chunk size 49 thread chunk size 2

//CHECK-NEXT: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4 B0T5 B0T5
//CHECK-SAME: B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9 B0T10 B0T10 B0T11 B0T11
//CHECK-SAME: B0T12 B0T12 B0T13 B0T13 B0T14 B0T14 B0T15 B0T15 B0T16 B0T16
//CHECK-SAME: B0T17 B0T17 B0T18 B0T18 B0T19 B0T19 B0T20 B0T20 B0T21 B0T21
//CHECK-SAME: B0T22 B0T22 B0T23 B0T23 B0T24
//CHECK-SAME: B1T0 B1T0 B1T1 B1T1 B1T2 B1T2 B1T3 B1T3 B1T4 B1T4 B1T5 B1T5
//CHECK-SAME: B1T6 B1T6 B1T7 B1T7 B1T8 B1T8 B1T9 B1T9 B1T10 B1T10 B1T11 B1T11
//CHECK-SAME: B1T12 B1T12 B1T13 B1T13 B1T14 B1T14 B1T15 B1T15 B1T16 B1T16
//CHECK-SAME: B1T17 B1T17 B1T18 B1T18 B1T19 B1T19 B1T20 B1T20 B1T21 B1T21
//CHECK-SAME: B1T22 B1T22 B1T23 B1T23 B1T24
//CHECK-SAME: B2T0 B2T0

//CHECK:      omp target teams distribute parallel for
//CHECK-SAME: block chunk size 29 thread chunk size default

//CHECK-NEXT: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8
//CHECK-SAME: B1T0 B1T1 B1T2 B1T3 B1T4 B1T5 B1T6 B1T7 B1T8 B1T9
//CHECK-SAME: B1T0 B1T1 B1T2 B1T3 B1T4 B1T5 B1T6 B1T7 B1T8 B1T9
//CHECK-SAME: B1T0 B1T1 B1T2 B1T3 B1T4 B1T5 B1T6 B1T7 B1T8
//CHECK-SAME: B2T0 B2T1 B2T2 B2T3 B2T4 B2T5 B2T6 B2T7 B2T8 B2T9
//CHECK-SAME: B2T0 B2T1 B2T2 B2T3 B2T4 B2T5 B2T6 B2T7 B2T8 B2T9
//CHECK-SAME: B2T0 B2T1 B2T2 B2T3 B2T4 B2T5 B2T6 B2T7 B2T8
//CHECK-SAME: B3T0 B3T1 B3T2 B3T3 B3T4 B3T5 B3T6 B3T7 B3T8 B3T9
//CHECK-SAME: B3T0 B3T1 B3T2

//CHECK:      omp target teams distribute parallel for
//CHECK-SAME: block chunk size 101 thread chunk size default

//CHECK-NEXT: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9
//CHECK-SAME: B0T0 B0T1 B0T2 B0T3 B0T4 B0T5 B0T6 B0T7 B0T8 B0T9

//CHECK:      omp target teams distribute parallel for
//CHECK-SAME: default block chunk size thread chunk size 101

//CHECK-NEXT: B0T0 B0T0 B0T0 B0T0 B0T0 B0T0 B0T0 B0T0 B0T0 B0T0
//CHECK-SAME: B1T0 B1T0 B1T0 B1T0 B1T0 B1T0 B1T0 B1T0 B1T0 B1T0
//CHECK-SAME: B2T0 B2T0 B2T0 B2T0 B2T0 B2T0 B2T0 B2T0 B2T0 B2T0
//CHECK-SAME: B3T0 B3T0 B3T0 B3T0 B3T0 B3T0 B3T0 B3T0 B3T0 B3T0
//CHECK-SAME: B4T0 B4T0 B4T0 B4T0 B4T0 B4T0 B4T0 B4T0 B4T0 B4T0
//CHECK-SAME: B5T0 B5T0 B5T0 B5T0 B5T0 B5T0 B5T0 B5T0 B5T0 B5T0
//CHECK-SAME: B6T0 B6T0 B6T0 B6T0 B6T0 B6T0 B6T0 B6T0 B6T0 B6T0
//CHECK-SAME: B7T0 B7T0 B7T0 B7T0 B7T0 B7T0 B7T0 B7T0 B7T0 B7T0
//CHECK-SAME: B8T0 B8T0 B8T0 B8T0 B8T0 B8T0 B8T0 B8T0 B8T0 B8T0
//CHECK-SAME: B0T0 B0T0 B0T0 B0T0 B0T0 B0T0 B0T0 B0T0 B0T0 B0T0
