// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// REQUIRES: gpu

#include "omp.h"
#include <stdlib.h>
#include <string.h>

#include "callbacks.h"
#include "register_non_emi.h"

#define N 1024

int main(int argc, char **argv) {
  int *h_a;
  int *d_a;

  h_a = (int *)malloc(N * sizeof(int));
  memset(h_a, 0, N);

  d_a = (int *)omp_target_alloc(N * sizeof(int), omp_get_default_device());

  omp_target_associate_ptr(h_a, d_a, N * sizeof(int), 0,
                           omp_get_default_device());
  omp_target_disassociate_ptr(h_a, omp_get_default_device());

  omp_target_free(d_a, omp_get_default_device());
  free(h_a);

  return 0;
}

// clang-format off
/// CHECK: Callback Init:
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=1
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=5
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=6
/// CHECK: Callback DataOp: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] optype=4
/// CHECK: Callback Fini:
