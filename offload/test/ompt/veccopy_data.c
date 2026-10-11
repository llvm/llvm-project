// clang-format off
// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: gpu
// clang-format on

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

// clang-format off
/// CHECK-NOT: Callback Target EMI:
/// CHECK-NOT: device_num=-1
/// CHECK: Callback Init:
/// CHECK: Callback Load:
/// CHECK: Callback Target EMI: kind=ompt_target_enter_data endpoint=ompt_scope_begin
/// CHECK-NOT: device_num=-1
/// CHECK-NOT: code=(nil)
/// CHECK: code=[[CODE1:.*]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_alloc
/// CHECK: code=[[CODE1]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_alloc
/// CHECK-NOT: dest=(nil)
/// CHECK: code=[[CODE1]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_to_device
/// CHECK: code=[[CODE1]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_to_device
/// CHECK: code=[[CODE1]]
/// CHECK: Callback Target EMI: kind=ompt_target_enter_data endpoint=ompt_scope_end
/// CHECK-NOT: device_num=-1
/// CHECK: code=[[CODE1]]
/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_begin
/// CHECK-NOT: device_num=-1
/// CHECK-NOT: code=(nil)
/// CHECK: code=[[CODE2:.*]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_alloc
/// CHECK: code=[[CODE2]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_alloc
/// CHECK-NOT: dest=(nil)
/// CHECK: code=[[CODE2]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_to_device
/// CHECK: code=[[CODE2]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_to_device
/// CHECK: code=[[CODE2]]
/// CHECK: Callback Submit EMI: endpoint=ompt_scope_begin  req_num_teams=1
/// CHECK: Callback Submit EMI: endpoint=ompt_scope_end  req_num_teams=1
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device
/// CHECK: code=[[CODE2]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device
/// CHECK: code=[[CODE2]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_delete
/// CHECK: code=[[CODE2]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_delete
/// CHECK: code=[[CODE2]]
/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_end
/// CHECK-NOT: device_num=-1
/// CHECK: code=[[CODE2]]
/// CHECK: Callback Target EMI: kind=ompt_target_exit_data endpoint=ompt_scope_begin
/// CHECK-NOT: device_num=-1
/// CHECK-NOT: code=(nil)
/// CHECK: code=[[CODE3:.*]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device
/// CHECK: code=[[CODE3]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device
/// CHECK: code=[[CODE3]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_delete
/// CHECK: code=[[CODE3]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_delete
/// CHECK: code=[[CODE3]]
/// CHECK: Callback Target EMI: kind=ompt_target_exit_data endpoint=ompt_scope_end
/// CHECK-NOT: device_num=-1
/// CHECK: code=[[CODE3]]
/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_begin
/// CHECK-NOT: device_num=-1
/// CHECK-NOT: code=(nil)
/// CHECK: code=[[CODE4:.*]]
/// CHECK: Callback Submit EMI: endpoint=ompt_scope_begin  req_num_teams=1
/// CHECK: Callback Submit EMI: endpoint=ompt_scope_end  req_num_teams=1
/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_end
/// CHECK-NOT: device_num=-1
/// CHECK: code=[[CODE4]]
/// CHECK: Callback Target EMI: kind=ompt_target_update endpoint=ompt_scope_begin
/// CHECK-NOT: device_num=-1
/// CHECK-NOT: code=(nil)
/// CHECK: code=[[CODE5:.*]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device
/// CHECK: code=[[CODE5]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device
/// CHECK: code=[[CODE5]]
/// CHECK: Callback Target EMI: kind=ompt_target_update endpoint=ompt_scope_end
/// CHECK-NOT: device_num=-1
/// CHECK: code=[[CODE5]]
/// CHECK: Callback Fini:
