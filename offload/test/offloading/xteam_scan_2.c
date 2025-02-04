// clang-format off
// This test verifies the output of inclusive and exclusive scan computed using the segmented variant 
// of the Xteam Scan Kernel. 
// It verifies that the reduction kernel is of Xteam-Scan type 
// and is launched with 85x256 and 85x512 combinations for teamsXthrds. 
// It also verifies the output without the num_teams() and num_threads() clauses.
// 

// RUN: %libomptarget-compile-generic -fopenmp-target-ignore-env-vars -fopenmp-target-xteam-scan -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state -lm -latomic
// RUN: env LIBOMPTARGET_KERNEL_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// RUN: %libomptarget-compile-generic -fopenmp-target-ignore-env-vars -fopenmp-target-xteam-scan -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state -lm -latomic -DNUM_THREADS=512
// RUN: env LIBOMPTARGET_KERNEL_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=CHECK-512WGSize

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

// clang-format on
#include <stdio.h>
#include <stdlib.h>
#ifndef NUM_TEAMS
#define NUM_TEAMS 85
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 256
#endif

#define N 2000000

int test_with_clauses() {
  int *in = (int*)malloc(sizeof(int) * N);
  int *out1 = (int*)malloc(sizeof(int) * N);

  for (int i = 0; i < N; i++) {
    in[i] = 10;
    out1[i] = 0;
  }

  int sum1;
  sum1 = 0;

#pragma omp target teams distribute parallel for reduction(inscan, +:sum1) map(tofrom: in[0:N], out1[0:N]) num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
  for (int i = 0; i < N; i++) {
    sum1 += in[i]; // input phase
#pragma omp scan inclusive(sum1)
    out1[i] = sum1; // scan phase
  }

  int checksum = 0;
  for (int i = 0; i < N; i++) {
    checksum += in[i];
    if (checksum != out1[i]) {
      printf("Inclusive Scan: Failure. Wrong Result at %d. Expecting: %d, Got: "
             "%d! Exiting...\n",
             i, checksum, out1[i]);
      return 1;
    }
  }
  free(out1);
  printf("Inclusive Scan: Success!\n");

  int sum2;
  sum2 = 0;
  int *out2 = (int *)malloc(sizeof(int) * N);

#pragma omp target teams distribute parallel for reduction(inscan, +:sum2) map(tofrom: in[0:N], out2[0:N]) num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
  for (int i = 0; i < N; i++) {
    out2[i] = sum2; // scan phase
#pragma omp scan exclusive(sum2)
    sum2 += in[i]; // input phase
  }

  checksum = 0;
  for (int i = 0; i < N; i++) {
    if (checksum != out2[i]) {
      printf("Exclusive Scan: Failure. Wrong Result at %d. Expecting: %d, Got: "
             "%d! Exiting...\n",
             i, checksum, out2[i]);
      return 1;
    }
    checksum += in[i];
  }
  free(in);
  free(out2);
  printf("Exclusive Scan: Success!\n");

  return 0;
}

int test_without_clauses() {
  int *in = (int*)malloc(sizeof(int) * N);
  int *out1 = (int*)malloc(sizeof(int) * N);

  for (int i = 0; i < N; i++) {
    in[i] = 10;
    out1[i] = 0;
  }

  int sum1;
  sum1 = 0;

#pragma omp target teams distribute parallel for reduction(inscan, +:sum1) map(tofrom: in[0:N], out1[0:N])
  for (int i = 0; i < N; i++) {
    sum1 += in[i]; // input phase
#pragma omp scan inclusive(sum1)
    out1[i] = sum1; // scan phase
  }

  int checksum = 0;
  for (int i = 0; i < N; i++) {
    checksum += in[i];
    if (checksum != out1[i]) {
      printf("Inclusive Scan: Failure. Wrong Result at %d. Expecting: %d, Got: "
             "%d! Exiting...\n",
             i, checksum, out1[i]);
      return 1;
    }
  }
  free(out1);
  printf("Inclusive Scan: Success!\n");

  int sum2;
  sum2 = 0;
  int *out2 = (int *)malloc(sizeof(int) * N);

#pragma omp target teams distribute parallel for reduction(inscan, +:sum2) map(tofrom: in[0:N], out2[0:N])
  for (int i = 0; i < N; i++) {
    out2[i] = sum2; // scan phase
#pragma omp scan exclusive(sum2)
    sum2 += in[i]; // input phase
  }

  checksum = 0;
  for (int i = 0; i < N; i++) {
    if (checksum != out2[i]) {
      printf("Exclusive Scan: Failure. Wrong Result at %d. Expecting: %d, Got: "
             "%d! Exiting...\n",
             i, checksum, out2[i]);
      return 1;
    }
    checksum += in[i];
  }
  free(in);
  free(out2);
  printf("Exclusive Scan: Success!\n");

  return 0;
}

int main() {
  int r1 = test_with_clauses();
  int r2 = test_without_clauses();
  return r1 || r2;
}

// clang-format off

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 85X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*]]_with_clauses_l49

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 85X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_with_clauses_l49_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 85X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*]]_with_clauses_l73

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 85X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_with_clauses_l73_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: n:__omp_offloading_[[MANGLED:.*]]_without_clauses_l109

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: n:__omp_offloading_[[MANGLED]]_without_clauses_l109_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: n:__omp_offloading_[[MANGLED:.*]]_without_clauses_l133

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: n:__omp_offloading_[[MANGLED]]_without_clauses_l133_1

/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK-512WGSize: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK-512WGSize: args:10 teamsXthrds:( 85X 512)
/// CHECK-512WGSize: n:__omp_offloading_[[MANGLED:.*]]_with_clauses_l49

/// CHECK-512WGSize: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK-512WGSize: args:10 teamsXthrds:( 85X 512)
/// CHECK-512WGSize: n:__omp_offloading_[[MANGLED]]_with_clauses_l49_1

/// CHECK-512WGSize: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK-512WGSize: args:10 teamsXthrds:( 85X 512)
/// CHECK-512WGSize: n:__omp_offloading_[[MANGLED:.*]]_with_clauses_l73

/// CHECK-512WGSize: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK-512WGSize: args:10 teamsXthrds:( 85X 512)
/// CHECK-512WGSize: n:__omp_offloading_[[MANGLED]]_with_clauses_l73_1

/// CHECK-512WGSize: Inclusive Scan: Success!
/// CHECK-512WGSize: Exclusive Scan: Success!