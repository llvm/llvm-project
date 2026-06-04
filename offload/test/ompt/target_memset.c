// clang-format off
// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: gpu
// clang-format on

/*
 * Verify that for the target OpenMP APIs, the return address is non-null and
 * distinct.
 */

#include <omp.h>
#include <stdlib.h>

#include "callbacks.h"
#include "register_non_emi.h"

int main() {
  int d = omp_get_default_device();
  int id = omp_get_initial_device();
  int q[128], i;
  void *p;
  void *result;

  if (d < 0 || d >= omp_get_num_devices())
    d = id;

  p = omp_target_alloc(130 * sizeof(int), d);
  if (p == NULL)
    return 0;

  for (i = 0; i < 128; i++)
    q[i] = i;

  result = omp_target_memset(p, 0, 130 * sizeof(int), d);
  if (result != p) {
    abort();
  }

  int q2[128];
  for (i = 0; i < 128; ++i)
    q2[i] = i;
  if (omp_target_memcpy_async(q2, p, 128 * sizeof(int), 0, sizeof(int), id, d,
                              0, NULL))
    abort();

#pragma omp taskwait

  for (i = 0; i < 128; ++i)
    if (q2[i] != 0)
      abort();

  omp_target_free(p, d);

  return 0;
}

// clang-format off
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_alloc
/// CHECK-SAME: src_device_num=[[HOST:-1]]
/// CHECK-SAME: dest_device_num=[[DEVICE:[0-9]+]]
/// CHECK-NOT: code=(nil)
/// CHECK: code=[[CODE1:0x[0-f]+]]
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_memset
/// CHECK-SAME: src_device_num=[[HOST]] {{.+}} dest_device_num=[[DEVICE]]
/// CHECK-NOT: code=(nil)
/// CHECK-NOT: code=[[CODE1]]
/// CHECK: code=[[CODE2:0x[0-f]+]]
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_transfer_from_device
/// CHECK-SAME: src_device_num=[[DEVICE]] {{.+}} dest_device_num=[[HOST]]
/// CHECK-NOT: code=(nil)
/// CHECK-NOT: code=[[CODE2]]
/// CHECK: code=[[CODE3:0x[0-f]+]]
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=ompt_target_data_delete
/// CHECK-NOT: code=(nil)
/// CHECK-NOT: code=[[CODE3]]
