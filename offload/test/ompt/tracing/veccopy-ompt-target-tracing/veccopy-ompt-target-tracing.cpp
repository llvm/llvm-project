// clang-format off
// RUN: %libomptarget-compilexx-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: amdgpu
// clang-format on
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

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

// > OMPT device tracing related checks below. <

// Note: This test will allocate one buffer, big enough to hold all trace
//       records, hence there will be only one allocation.


// Note: Split checks for record address and content. That way we do not imply
//       any order. Records may / will occur interleaved.

// Note: These addresses will only occur once. They are only captured to
//       indicate their existence.

/// CHECK-DAG: type=8 (Target task)
/// CHECK-DAG: type=9 (Target data op)

// Note: ADDRX_01 may not trigger a final callback.
// Note: ADDRX_01 may not be deallocated.

/// CHECK-NOT: host_op_id=0x0
