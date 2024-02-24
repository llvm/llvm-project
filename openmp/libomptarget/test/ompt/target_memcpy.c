// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

/*
 * Verify that for the target OpenMP APIs, the return address is non-null and
 * distinct.
 */

#include <omp.h>
#include <stdlib.h>

#include "callbacks.h"
#include "register_non_emi.h"

int main() {
  int dev = omp_get_default_device();
  int host = omp_get_initial_device();

  int host_var1 = 42;
  int host_var2 = 0;
  void *dev_ptr = NULL;

  // Allocate space on the device
  dev_ptr = omp_target_alloc(sizeof(int), dev);
  if (dev_ptr == NULL)
    abort();

  // H2D transfer
  if (omp_target_memcpy(dev_ptr, &host_var1, sizeof(int), 0, 0, dev, host))
    abort();

  // D2H transfer
  if (omp_target_memcpy(&host_var2, dev_ptr, sizeof(int), 0, 0, host, dev))
    abort();

  // Free the device location
  omp_target_free(dev_ptr, dev);

  // Both host variables should have the same value.
  return host_var1 != host_var2;
}

// clang-format off
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=1
/// CHECK-NOT: code=(nil)
/// CHECK: code=[[CODE1:.*]]
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=2
/// CHECK-NOT: code=(nil)
/// CHECK-NOT: code=[[CODE1]]
/// CHECK: code=[[CODE2:.*]]
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=3
/// CHECK-NOT: code=(nil)
/// CHECK-NOT: code=[[CODE2]]
/// CHECK: code=[[CODE3:.*]]
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=4
/// CHECK-NOT: code=(nil)
/// CHECK-NOT: code=[[CODE3]]
