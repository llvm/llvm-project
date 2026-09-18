// clang-format off
// RUN: %libomptarget-compilexx-generic && env LIBOMPTARGET_OMPT_FLUSH_ON_SHUTDOWN=false %libomptarget-run-generic | %fcheck-generic
// REQUIRES: ompt
// REQUIRES: amdgpu
// clang-format on
/*
 * This test is run with LIBOMPTARGET_OMPT_FLUSH_ON_SHUTDOWN=false and
 * ompt_flush_trace is not invoked by the user/tool. The intention is to check
 * whether trace records are properly flushed as a buffer fills up. Currently,
 * the buffer size in callbacks.h implies that every buffer holds 2 trace
 * records. With this assumption, there should be 22 trace records. The last
 * trace record is not flushed because the last buffer is not yet full.
 */
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

/// CHECK-NOT: host_op_id=0x0

/// CHECK: rec=
/// CHECK-NOT: host_op_id=0x0
