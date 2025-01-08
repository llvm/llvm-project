// clang-format off
// RUN: %libomptarget-compile-generic && %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

// REQUIRES: amdgcn-amd-amdhsa

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int ordered_example(int lb, int ub, int stride, int nteams) {
  int i;
  int size = (ub - lb) / stride;
  double *output = (double *)malloc(size * sizeof(double));

#pragma omp target teams map(from : output[0 : size]) num_teams(nteams)        \
    thread_limit(128)
#pragma omp parallel for ordered schedule(dynamic)
  for (i = lb; i < ub; i += stride) {
#pragma omp ordered
    { output[(i - lb) / stride] = omp_get_wtime(); }
  }

  // verification
  for (int j = 0; j < size; j++) {
    for (int jj = j + 1; jj < size; jj++) {
      if (output[j] > output[jj]) {
        printf("Fail to schedule in order.\n");
        free(output);
        return 1;
      }
    }
  }

  free(output);

  printf("test ordered OK\n");

  return 0;
}

int NO_order_example(int lb, int ub, int stride, int nteams) {
  int i;
  int size = (ub - lb) / stride;
  double *output = (double *)malloc(size * sizeof(double));

#pragma omp target teams map(from : output[0 : size]) num_teams(nteams)        \
    thread_limit(128)
#pragma omp parallel for schedule(dynamic)
  for (i = lb; i < ub; i += stride) {
    output[(i - lb) / stride] = omp_get_wtime();
  }

  // verification
  for (int j = 0; j < size; j++) {
    for (int jj = j + 1; jj < size; jj++) {
      if (output[j] > output[jj]) {
        printf("Fail to schedule in order.\n");
        free(output);
        return 1;
      }
    }
  }

  free(output);

  printf("test no order OK\n");

  return 0;
}

int main() {
  // CHECK: test no order OK
  NO_order_example(0, 10, 1, 8);
  // CHECK: test ordered OK
  return ordered_example(0, 10, 1, 8);
}
