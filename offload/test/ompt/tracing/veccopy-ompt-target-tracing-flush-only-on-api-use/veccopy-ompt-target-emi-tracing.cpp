// clang-format off
// RUN: %libomptarget-compilexx-generic && env LIBOMPTARGET_OMPT_FLUSH_ON_SHUTDOWN=false LIBOMPTARGET_OMPT_FLUSH_ON_BUFFER_FULL=false %libomptarget-run-generic | %fcheck-generic
// REQUIRES: ompt
// REQUIRES: amdgpu
// clang-format on
/*
 * This test is run with LIBOMPTARGET_OMPT_FLUSH_ON_SHUTDOWN=false and
 * LIBOMPTARGET_OMPT_FLUSH_ON_BUFFER_FULL=false while doing explicit flushing
 * using ompt_flush_trace. The intention is to check whether trace records are
 * properly flushed when the program/tool uses the ompt_flush_trace API. There
 * should be 23 trace records returned to the tool.
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

  for (auto Dev : *DeviceMapPtr)
    flush_trace(Dev);

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  for (auto Dev : *DeviceMapPtr)
    flush_trace(Dev);

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

/// CHECK-DAG: Success

/// The user calls flush before printing success, so 
/// no more records should be returned here.

/// CHECK-NOT: host_op_id=0x0
