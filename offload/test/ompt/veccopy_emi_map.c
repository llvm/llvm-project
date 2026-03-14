// clang-format off
// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: gpu
// clang-format on

/*
 * Example OpenMP program that shows that map-EMI callbacks are not supported.
 */

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#include "callbacks.h"
#include "register_emi_map.h"

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
/// CHECK: 0: Could not register callback 'ompt_callback_target_map_emi'
/// CHECK: Callback Init:
/// CHECK: Callback Load:
/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_begin
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_alloc
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_alloc
/// CHECK-NOT: dest=(nil)
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_to_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_to_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_alloc
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_alloc
/// CHECK-NOT: dest=(nil)
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_to_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_to_device
/// CHECK: Callback Submit EMI: endpoint=ompt_scope_begin req_num_teams=1
/// CHECK: Callback Submit EMI: endpoint=ompt_scope_end req_num_teams=1
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_delete
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_delete
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_delete
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_delete
/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_end
/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_begin
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_alloc
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_alloc
/// CHECK-NOT: dest=(nil)
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_to_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_to_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_alloc
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_alloc
/// CHECK-NOT: dest=(nil)
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_to_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_to_device
/// CHECK: Callback Submit EMI: endpoint=ompt_scope_begin req_num_teams=0
/// CHECK: Callback Submit EMI: endpoint=ompt_scope_end req_num_teams=0
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_delete
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_delete
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_delete
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_delete
/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_end
/// CHECK: Callback Fini:
