// RUN: %libomptarget-compile-generic && %libomptarget-run-generic
// RUN: %libomptarget-compileopt-generic && %libomptarget-run-generic

#include <stdio.h>
#include <stdlib.h>

int main() {
  long unsigned *DP = 0;
  int N = 128;
  int Threads = 128;
  int Teams = 440;

  // Allocate ~55MB on the device.
#pragma omp target map(from : DP)
  DP = (long unsigned *)malloc(sizeof(long unsigned) * N * Threads * Teams);

#pragma omp target teams distribute parallel for num_teams(Teams)              \
    thread_limit(Threads) is_device_ptr(DP)
  for (int i = 0; i < Threads * Teams; ++i) {
    for (int j = 0; j < N; ++j) {
      DP[i * N + j] = i + j;
    }
  }

  long unsigned s = 0;
#pragma omp target teams distribute parallel for num_teams(Teams)              \
    thread_limit(Threads) reduction(+ : s)
  for (int i = 0; i < Threads * Teams; ++i) {
    for (int j = 0; j < N; ++j) {
      s += DP[i * N + j];
    }
  }

  // CHECK: Sum: 203458478080
  printf("Sum: %li\n", s);
  return 0;
}
