// clang-format off
// RUN: env LIBOMP_NUM_HIDDEN_HELPER_THREADS=1 %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// clang-format on

#include <inttypes.h>
#include <omp-tools.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "register_with_host.h"

#define N 1000000
#define M 1000

int main() {
  float *x = malloc(N * sizeof(float));
  float *y = malloc(N * sizeof(float));
  float *a = malloc(N * sizeof(float));
  float *b = malloc(N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1;
    y[i] = 1;
    a[i] = 1;
    b[i] = 1;
  }

#pragma omp target teams distribute parallel for nowait map(to : x[0 : N])     \
    map(from : y[0 : N])
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      y[i] += 3 * x[i];
    }
  }

#pragma omp target teams distribute parallel for nowait map(to : a[0 : N])     \
    map(from : b[0 : N])
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      b[i] += 3 * a[i];
    }
  }

#pragma omp taskwait

  printf("%f, %f, %f, %f\n", x[0], y[0], a[0], b[0]);

  free(x);
  free(y);
  free(a);
  free(b);
  return 0;
}

// clang-format off
/// CHECK-NOT: target_task_data=(nil) (0x0)
/// CHECK-NOT: target_data=(nil) (0x0)
