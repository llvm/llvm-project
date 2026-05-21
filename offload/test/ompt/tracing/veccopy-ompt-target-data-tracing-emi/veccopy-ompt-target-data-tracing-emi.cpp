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

#define N 100000

#pragma omp declare target
int c[N];
#pragma omp end declare target

int main() {
  int a[N];
  int b[N];

  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

  for (i = 0; i < N; i++)
    c[i] = 0;

#pragma omp target enter data map(to : a)
#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }
#pragma omp target exit data map(from : a)

  for (auto Dev : *DeviceMapPtr)
    flush_trace(Dev);

#pragma omp target parallel for map(alloc : c)
  {
    for (int j = 0; j < N; j++)
      c[j] = 2 * j + 1;
  }
#pragma omp target update from(c) nowait
#pragma omp barrier

  for (auto Dev : *DeviceMapPtr)
    flush_trace(Dev);

  int rc = 0;
  for (i = 0; i < N; i++) {
    if (a[i] != i) {
      rc++;
      printf("Wrong value: a[%d]=%d\n", i, a[i]);
    }
  }

  for (i = 0; i < N; i++) {
    if (c[i] != 2 * i + 1) {
      rc++;
      printf("Wrong value: c[%d]=%d\n", i, c[i]);
    }
  }

  if (!rc)
    printf("Success\n");

  return rc;
}

// clang-format off

// > OMPT device callback related checks below. <

/// CHECK-NOT: Callback Target EMI:
/// CHECK-NOT: device_num=-1

/// CHECK: Callback Init:
/// CHECK: Callback Load:

/// CHECK-DAG: Callback Target EMI: kind=2 endpoint=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=2
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=2
/// CHECK-DAG: Callback Target EMI: kind=2 endpoint=2

/// CHECK-DAG: Callback Target EMI: kind=1 endpoint=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=2
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=2
/// CHECK-DAG: Callback Submit EMI: endpoint=1 req_num_teams=1
/// CHECK-DAG: Callback Submit EMI: endpoint=2 req_num_teams=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=3
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=3
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=4
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=4
/// CHECK-DAG: Callback Target EMI: kind=1 endpoint=2
/// CHECK-DAG: Callback Target EMI: kind=3 endpoint=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=3
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=3
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=4
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=4
/// CHECK-DAG: Callback Target EMI: kind=3 endpoint=2

/// CHECK-DAG: Callback Target EMI: kind=1 endpoint=1
/// CHECK-DAG: Callback Submit EMI: endpoint=1 req_num_teams=1
/// CHECK-DAG: Callback Submit EMI: endpoint=2 req_num_teams=1
/// CHECK-DAG: Callback Target EMI: kind=1 endpoint=2
/// CHECK-DAG: Callback Target EMI: kind=4 endpoint=1
/// CHECK-DAG: Callback DataOp EMI: endpoint=1 optype=3
/// CHECK-DAG: Callback DataOp EMI: endpoint=2 optype=3
/// CHECK-DAG: Callback Target EMI: kind=4 endpoint=2

/// CHECK-DAG: Success
/// CHECK-DAG: Callback Fini:

// > OMPT device tracing related checks below. <



// Note: Split checks for record address and content. That way we do not imply
//       any order. Records 01-06 and 12-17 occur interleaved and belong to the
//       first target region. 07-11 occur interleaved with 18-22 and belong to
//       the second target region.

// Note: These addresses will only occur once. They are only captured to
//       indicate their existence.

/// CHECK-DAG: type=8 (Target task)
/// CHECK-DAG: type=9 (Target data op)

// Note: ADDRX_11 may not trigger a final callback.

// Note: ADDRX_11 may not be deallocated.

/// CHECK-NOT: host_op_id=0x0
