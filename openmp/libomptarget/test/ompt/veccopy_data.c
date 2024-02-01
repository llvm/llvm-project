// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

/*
 * Example OpenMP program that registers EMI callbacks.
 * Explicitly testing for an initialized device num and
 * #pragma omp target [data enter / data exit / update]
 * The latter with the addition of a nowait clause.
 */

#include <omp.h>
#include <stdio.h>

#include "callbacks.h"
#include "register_emi.h"

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

#pragma omp target parallel for map(alloc : c)
  {
    for (int j = 0; j < N; j++)
      c[j] = 2 * j + 1;
  }
#pragma omp target update from(c) nowait
#pragma omp barrier

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

/// CHECK-NOT: Callback Target EMI:
/// CHECK-NOT: device_num=-1
/// CHECK: Callback Init:
/// CHECK: Callback Load:
/// CHECK: Callback Target EMI: kind=2 endpoint=1
/// CHECK-NOT: device_num=-1
/// CHECK: Callback DataOp EMI: endpoint=1 optype=1
/// CHECK: Callback DataOp EMI: endpoint=2 optype=1
/// CHECK-NOT: dest=(nil)
/// CHECK: Callback DataOp EMI: endpoint=1 optype=2
/// CHECK: Callback DataOp EMI: endpoint=2 optype=2
/// CHECK: Callback Target EMI: kind=2 endpoint=2
/// CHECK-NOT: device_num=-1
/// CHECK: Callback Target EMI: kind=1 endpoint=1
/// CHECK-NOT: device_num=-1
/// CHECK: Callback DataOp EMI: endpoint=1 optype=1
/// CHECK: Callback DataOp EMI: endpoint=2 optype=1
/// CHECK-NOT: dest=(nil)
/// CHECK: Callback DataOp EMI: endpoint=1 optype=2
/// CHECK: Callback DataOp EMI: endpoint=2 optype=2
/// CHECK: Callback Submit EMI: endpoint=1  req_num_teams=1
/// CHECK: Callback Submit EMI: endpoint=2  req_num_teams=1
/// CHECK: Callback DataOp EMI: endpoint=1 optype=3
/// CHECK: Callback DataOp EMI: endpoint=2 optype=3
/// CHECK: Callback DataOp EMI: endpoint=1 optype=4
/// CHECK: Callback DataOp EMI: endpoint=2 optype=4
/// CHECK: Callback Target EMI: kind=1 endpoint=2
/// CHECK-NOT: device_num=-1
/// CHECK: Callback Target EMI: kind=3 endpoint=1
/// CHECK-NOT: device_num=-1
/// CHECK: Callback DataOp EMI: endpoint=1 optype=3
/// CHECK: Callback DataOp EMI: endpoint=2 optype=3
/// CHECK: Callback DataOp EMI: endpoint=1 optype=4
/// CHECK: Callback DataOp EMI: endpoint=2 optype=4
/// CHECK: Callback Target EMI: kind=3 endpoint=2
/// CHECK-NOT: device_num=-1
/// CHECK: Callback Target EMI: kind=1 endpoint=1
/// CHECK-NOT: device_num=-1
/// CHECK: Callback Submit EMI: endpoint=1  req_num_teams=1
/// CHECK: Callback Submit EMI: endpoint=2  req_num_teams=1
/// CHECK: Callback Target EMI: kind=1 endpoint=2
/// CHECK-NOT: device_num=-1
/// CHECK: Callback Target EMI: kind=4 endpoint=1
/// CHECK-NOT: device_num=-1
/// CHECK: Callback DataOp EMI: endpoint=1 optype=3
/// CHECK: Callback DataOp EMI: endpoint=2 optype=3
/// CHECK: Callback Target EMI: kind=4 endpoint=2
/// CHECK-NOT: device_num=-1
/// CHECK: Callback Fini:
