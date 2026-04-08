// clang-format off
// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: gpu
// clang-format on

/*
 * Example OpenMP program that shows that map callbacks are not supported.
 */

#include <omp.h>
#include <stdio.h>

#include "callbacks.h"
#include "register_non_emi_map.h"

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
/// CHECK: 0: Could not register callback 'ompt_callback_target_map'
/// CHECK: Callback Init:
/// CHECK: Callback Load:
/// CHECK: Callback Target: target_id=[[TARGET_ID:[0-9]+]] kind=ompt_target endpoint=ompt_scope_begin
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_alloc
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_transfer_to_device
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_alloc
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_transfer_to_device
/// CHECK: Callback Submit: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] req_num_teams=1
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_delete
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_delete
/// CHECK: Callback Target: target_id=[[TARGET_ID:[0-9]+]] kind=ompt_target endpoint=ompt_scope_end

/// CHECK: Callback Target: target_id=[[TARGET_ID:[0-9]+]] kind=ompt_target endpoint=ompt_scope_begin
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_alloc
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_transfer_to_device
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_alloc
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_transfer_to_device
/// CHECK: Callback Submit: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] req_num_teams=0
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_transfer_from_device
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_delete
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_delete
/// CHECK: Callback Target: target_id=[[TARGET_ID:[0-9]+]] kind=ompt_target endpoint=ompt_scope_end
/// CHECK: Callback Fini:
