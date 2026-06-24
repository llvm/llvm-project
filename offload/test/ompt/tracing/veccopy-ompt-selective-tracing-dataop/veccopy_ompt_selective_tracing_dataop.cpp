// clang-format off
// RUN: %libomptarget-compilexx-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: amdgpu
// clang-format on
#include <assert.h>
#include <omp.h>
#include <stdio.h>

#include "callbacks.h"

// Map of devices traced
DeviceMapPtr_t DeviceMapPtr;

int main() {
  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  int rc = 0;
  for (i = 0; i < N; i++)
    if (a[i] != b[i]) {
      rc++;
      printf("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

// clang-format off

/// CHECK-NOT: rec={{0x[0-9a-fA-F]+}} type=8
/// CHECK: rec={{0x[0-9a-fA-F]+}} type=9
/// CHECK-NOT: rec={{0x[0-9a-fA-F]+}} type=10
/// CHECK: rec={{0x[0-9a-fA-F]+}} type=9
/// CHECK-NOT: rec={{0x[0-9a-fA-F]+}} type=8

/// CHECK-NOT: rec={{0x[0-9a-fA-F]+}} type=8
/// CHECK: rec={{0x[0-9a-fA-F]+}} type=9
/// CHECK-NOT: rec={{0x[0-9a-fA-F]+}} type=10
/// CHECK: rec={{0x[0-9a-fA-F]+}} type=9
/// CHECK-NOT: rec={{0x[0-9a-fA-F]+}} type=8
