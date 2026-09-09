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

  for (auto Dev : *DeviceMapPtr)
    flush_trace(Dev);

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



// Note: Split checks for record address and content. That way we do not imply
//       any order. Records 01-06 and 12-17 occur interleaved and belong to the
//       first target region. 07-11 occur interleaved with 18-22 and belong to
//       the second target region.

/// CHECK-DAG: type=8 (Target task)
/// CHECK-DAG: type=9 (Target data op)

// Note: These addresses will only occur once. They are only captured to
//       indicate their existence.


// Note: ADDRX_11 may not trigger a final callback.

// Note: ADDRX_11 may not be deallocated.

/// CHECK-DAG: Success

/// CHECK-NOT: host_op_id=0x0
