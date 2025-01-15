// clang-format off
// This test verifies the output of inclusive and exclusive scan computed using the Xteam Scan Kernel
// for various datatypes for the default teamsXthreads combination: 104x1024
// 

// RUN: %libomptarget-compile-generic -fopenmp-target-ignore-env-vars -fopenmp-target-xteam-scan -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state -lm -latomic
// RUN: env LIBOMPTARGET_KERNEL_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

// clang-format on
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define N 2000000

template<typename T>
void run_test() {
  T *in = (T*)malloc(sizeof(T) * N);
  T *out1 = (T*)malloc(sizeof(T) * N);  // For inclusive scan
  T *out2 = (T *)malloc(sizeof(T) * N); // For exclusive scan

  for (int i = 0; i < N; i++) {
    in[i] = 10;
    out1[i] = 0;
  }

  T sum1 = T(0);

#pragma omp target teams distribute parallel for reduction(inscan, +:sum1) map(tofrom: in[0:N], out1[0:N])
  for (int i = 0; i < N; i++) {
    sum1 += in[i]; // input phase
#pragma omp scan inclusive(sum1)
    out1[i] = sum1; // scan phase
  }

  T checksum = T(0);
  for (int i = 0; i < N; i++) {
    checksum += in[i];
    if (checksum != out1[i]) {
      printf("Inclusive Scan: Failure. Wrong Result at %d. Exiting...\n", i);
      return;
    }
  }
  free(out1);
  printf("Inclusive Scan: Success!\n");

  T sum2 = T(0);

#pragma omp target teams distribute parallel for reduction(inscan, +:sum2) map(tofrom: in[0:N], out2[0:N])
  for (int i = 0; i < N; i++) {
    out2[i] = sum2; // scan phase
#pragma omp scan exclusive(sum2)
    sum2 += in[i]; // input phase
  }

  checksum = T(0);
  for (int i = 0; i < N; i++) {
    if (checksum != out2[i]) {
      printf("Exclusive Scan: Failure. Wrong Result at %d. Exiting...\n", i);
      return;
    }
    checksum += in[i];
  }
  free(in);
  free(out2);
  printf("Exclusive Scan: Success!\n");
}

int main() {
  printf("Testing for datatype int\n");
  run_test<int>();
  
  printf("Testing for datatype uint32_t\n");
  run_test<uint32_t>();

  printf("Testing for datatype uint64_t\n");
  run_test<uint64_t>();

  printf("Testing for datatype long\n");
  run_test<long>();

  printf("Testing for datatype double\n");
  run_test<double>();
  
  printf("Testing for datatype float\n");
  run_test<float>();
  return 0;
}
// clang-format off

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:8200B
/// CHECK: n:__omp_offloading_[[MANGLED:.*i.*]]_l37

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l37_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:8200B
/// CHECK: n:__omp_offloading_[[MANGLED:.*i.*]]_l57

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l57_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:8200B
/// CHECK: n:__omp_offloading_[[MANGLED:.*j.*]]_l37

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l37_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:8200B
/// CHECK: n:__omp_offloading_[[MANGLED:.*j.*]]_l57

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l57_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:16400B
/// CHECK: n:__omp_offloading_[[MANGLED:.*m.*]]_l37

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l37_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:16400B
/// CHECK: n:__omp_offloading_[[MANGLED:.*m.*]]_l57

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l57_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:16400B
/// CHECK: n:__omp_offloading_[[MANGLED:.*l.*]]_l37

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l37_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:16400B
/// CHECK: n:__omp_offloading_[[MANGLED:.*l.*]]_l57

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l57_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:16400B
/// CHECK: n:__omp_offloading_[[MANGLED:.*d.*]]_l37

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l37_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:16400B
/// CHECK: n:__omp_offloading_[[MANGLED:.*d.*]]_l57

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l57_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:8200B
/// CHECK: n:__omp_offloading_[[MANGLED:.*f.*]]_l37

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l37_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:8200B
/// CHECK: n:__omp_offloading_[[MANGLED:.*f.*]]_l57

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args:10 teamsXthrds:( 104X1024)
/// CHECK: lds_usage:0B
/// CHECK: n:__omp_offloading_[[MANGLED]]_l57_1

/// CHECK: Testing for datatype int
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK: Testing for datatype uint32_t
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK: Testing for datatype uint64_t
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK: Testing for datatype long
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK: Testing for datatype double
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK: Testing for datatype float
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!