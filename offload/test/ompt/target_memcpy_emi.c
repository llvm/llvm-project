// clang-format off
// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: gpu
// clang-format on

/*
 * Verify all three data transfer directions: H2D, D2D and D2H
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

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

  printf("Freeing Memory on Device\n");
  free(HstPtr);
  omp_target_free(DevPtr, Device);

  return 0;
}

// clang-format off

/// CHECK: Callback Init:

/// CHECK: Allocating Memory on Device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_alloc
/// CHECK-SAME: src_device_num=[[HOST:[0-9]+]]
/// CHECK-SAME: dest_device_num=[[DEVICE:[0-9]+]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_alloc {{.+}} src_device_num=[[HOST]] {{.+}} dest_device_num=[[DEVICE]]

/// CHECK: Testing: Host to Device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_to_device {{.+}} src_device_num=[[HOST]] {{.+}} dest_device_num=[[DEVICE]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_to_device {{.+}} src_device_num=[[HOST]] {{.+}} dest_device_num=[[DEVICE]]

/// CHECK: Testing: Device to Device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device {{.+}} src_device_num=[[DEVICE]] {{.+}} dest_device_num=[[DEVICE]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device {{.+}} src_device_num=[[DEVICE]] {{.+}} dest_device_num=[[DEVICE]]

/// CHECK: Testing: Device to Host
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_transfer_from_device {{.+}} src_device_num=[[DEVICE]] {{.+}} dest_device_num=[[HOST]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_transfer_from_device {{.+}} src_device_num=[[DEVICE]] {{.+}} dest_device_num=[[HOST]]

/// CHECK: Checking Correctness

/// CHECK: Freeing Memory on Device
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin optype=ompt_target_data_delete {{.+}} src_device_num=[[DEVICE]]
/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end optype=ompt_target_data_delete {{.+}} src_device_num=[[DEVICE]]

/// CHECK: Callback Fini:

// clang-format on
