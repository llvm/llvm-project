// clang-format off
// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: gpu
// clang-format on

/*
 * Example OpenMP program that checks if EMI callbacks
 * correctly pass host_op_id between ompt_scope_begin and ompt_scope_end
 */

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#include "callbacks.h"
#include "register_emi.h"

int main(void) {
  int NumDevices = omp_get_num_devices();
  assert(NumDevices > 0 && "No device(s) present.");
  int Device = omp_get_default_device();
  int Host = omp_get_initial_device();
  // Note: Zero value depicts an OFFLOAD_SUCCESS
  int Status;

  printf("Allocating Memory on Device\n");
  int *DevPtr = (int *)omp_target_alloc(sizeof(int), Device);
  assert(DevPtr && "Could not allocate memory on device.");
  int *HstPtr = (int *)malloc(sizeof(int));
  *HstPtr = 42;

  printf("Testing: Host to Device\n");
  Status = omp_target_memcpy(DevPtr, HstPtr, sizeof(int), 0, 0, Device, Host);
  assert(Status == 0 && "H2D memory copy operation failed.\n");

  printf("Testing: Device to Device\n");
  Status = omp_target_memcpy(DevPtr, DevPtr, sizeof(int), 0, 0, Device, Device);
  assert(Status == 0 && "D2D memory copy operation failed.\n");

  printf("Testing: Device to Host\n");
  Status = omp_target_memcpy(HstPtr, DevPtr, sizeof(int), 0, 0, Host, Device);
  assert(Status == 0 && "D2H memory copy operation failed.\n");

  printf("Checking Correctness\n");
  assert(*HstPtr == 42);

  printf("Setting Device Memory to 0\n");
  int *DevMemsetPtr = omp_target_memset(DevPtr, 0, sizeof(int), Device);
  assert(DevMemsetPtr == DevPtr && "Memset returned incorrect pointer.\n");

  printf("Submitting a Kernel\n");
#pragma omp target
  *DevPtr = 42;

  printf("Freeing Memory on Device\n");
  free(HstPtr);
  omp_target_free(DevPtr, Device);

  return 0;
}

// clang-format off

/// CHECK: Callback Init:

/// CHECK: Allocating Memory on Device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_alloc {{.+}} host_op_id=[[HOSTOP1:0x[[:xdigit:]]+ \(0x[[:xdigit:]]+\)]] {{.+}}
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_alloc {{.+}} host_op_id=[[HOSTOP1]] {{.+}}

/// CHECK: Testing: Host to Device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_to_device {{.+}} host_op_id=[[HOSTOP2:0x[[:xdigit:]]+ \(0x[[:xdigit:]]+\)]] {{.+}}
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_to_device {{.+}} host_op_id=[[HOSTOP2]] {{.+}}

/// CHECK: Testing: Device to Device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device {{.+}} host_op_id=[[HOSTOP3:0x[[:xdigit:]]+ \(0x[[:xdigit:]]+\)]] {{.+}}
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device {{.+}} host_op_id=[[HOSTOP3]] {{.+}}

/// CHECK: Testing: Device to Host
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device {{.+}} host_op_id=[[HOSTOP4:0x[[:xdigit:]]+ \(0x[[:xdigit:]]+\)]] {{.+}}
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device {{.+}} host_op_id=[[HOSTOP4]] {{.+}}

/// CHECK: Checking Correctness

/// CHECK: Setting Device Memory to 0
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_memset {{.+}} host_op_id=[[HOSTOP5:0x[[:xdigit:]]+ \(0x[[:xdigit:]]+\)]] {{.+}}
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_memset {{.+}} host_op_id=[[HOSTOP5]] {{.+}}

/// CHECK: Submitting a Kernel
/// CHECK: Callback Submit EMI: endpoint=ompt_scope_begin {{.+}} host_op_id=[[HOSTOP6:0x[[:xdigit:]]+ \(0x[[:xdigit:]]+\)]]
/// CHECK: Callback Submit EMI: endpoint=ompt_scope_end {{.+}} host_op_id=[[HOSTOP6]]

/// CHECK: Freeing Memory on Device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_delete {{.+}} host_op_id=[[HOSTOP7:0x[[:xdigit:]]+ \(0x[[:xdigit:]]+\)]] {{.+}}
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_delete {{.+}} host_op_id=[[HOSTOP7]] {{.+}}

/// CHECK: Callback Fini:

// clang-format on
