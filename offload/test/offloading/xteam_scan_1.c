// clang-format off
// This test verifies the output of inclusive scan and exclusive scan computed using
// the Xteam Scan Kernel approach. It verifies that the reduction kernel is of 
// Xteam-Scan type and is launched with 250x256 and 100x512 combinations for teamsXthrds. 
// 

// RUN: %libomptarget-compile-generic -fopenmp-target-ignore-env-vars -fopenmp-target-xteam-scan -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state -lm -latomic
// RUN: env LIBOMPTARGET_KERNEL_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// RUN: %libomptarget-compile-generic -fopenmp-target-ignore-env-vars -fopenmp-target-xteam-scan -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state -lm -latomic -DNUM_THREADS=512 -DNUM_TEAMS=100
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
#ifndef NUM_TEAMS
#define NUM_TEAMS 250
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 256
#endif

#define N NUM_THREADS *NUM_TEAMS

int main() {
  int in[N], out1[N];

  for (int i = 0; i < N; i++) {
    in[i] = 10;
    out1[i] = 0;
  }

  int sum1;
  sum1 = 0;

#pragma omp target teams distribute parallel for reduction(inscan, +:sum1) map(tofrom: in, out1) num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
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

  printf("Inclusive Scan: Success!\n");

  int sum2 = 0;
  int out2[N];
#pragma omp target teams distribute parallel for reduction(inscan, +:sum2) map(tofrom: in, out2) num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
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
             i, checksum, out1[i]);
      return 1;
    }
    checksum += in[i];
  }

  printf("Exclusive Scan: Success!\n");

  return 0;
}
// clang-format off
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*]]_main_l45

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_main_l45_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_main_l67

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_main_l67_1
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK-512WGSize: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK-512WGSize: args: 9 teamsXthrds:( 100X 512)
/// CHECK-512WGSize: n:__omp_offloading_[[MANGLED:.*]]_main_l45

/// CHECK-512WGSize: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK-512WGSize: args: 9 teamsXthrds:( 100X 512)
/// CHECK-512WGSize: n:__omp_offloading_[[MANGLED]]_main_l45_1

/// CHECK-512WGSize: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK-512WGSize: args: 9 teamsXthrds:( 100X 512)
/// CHECK-512WGSize: n:__omp_offloading_[[MANGLED]]_main_l67

/// CHECK-512WGSize: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK-512WGSize: args: 9 teamsXthrds:( 100X 512)
/// CHECK-512WGSize: n:__omp_offloading_[[MANGLED]]_main_l67_1
/// CHECK-512WGSize: Inclusive Scan: Success!
/// CHECK-512WGSize: Exclusive Scan: Success!
