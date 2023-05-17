// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-oldDriver
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-oldDriver
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

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

/// CHECK: Could not register callback 'ompt_callback_device_initialize'
/// CHECK: Could not register callback 'ompt_callback_device_finalize'
/// CHECK: Could not register callback 'ompt_callback_device_load'
/// CHECK: Could not register callback 'ompt_callback_target_data_op_emi'
/// CHECK: Could not register callback 'ompt_callback_target_emi'
/// CHECK: Could not register callback 'ompt_callback_target_submit_emi'
/// CHECK: Could not register callback 'ompt_callback_target_map_emi'

/// CHECK: Success
