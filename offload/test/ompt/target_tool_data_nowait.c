// clang-format off
// RUN: env LIBOMP_NUM_HIDDEN_HELPER_THREADS=1 %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// clang-format on

#include <inttypes.h>
#include <omp-tools.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "register_with_host.h"

#define N 1000000
#define M 1000

int main() {
  float *x = malloc(N * sizeof(float));
  float *y = malloc(N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1;
    y[i] = 1;
  }

#pragma omp target enter data map(to : x[0 : N]) map(alloc : y[0 : N])         \
    nowait depend(inout : x)
#pragma omp target teams distribute parallel for nowait depend(inout : x)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      y[i] += 3 * x[i];
    }
  }

#pragma omp target teams distribute parallel for nowait depend(inout : x)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      y[i] += 3 * x[i];
    }
  }

#pragma omp target exit data map(release : x[0 : N]) map(from : y[0 : N])      \
    nowait depend(inout : x)
#pragma omp taskwait

  printf("%f, %f\n", x[0], y[0]);

  free(x);
  free(y);
  return 0;
}

// clang-format off
/// CHECK: ompt_event_initial_task_begin
/// CHECK-SAME: task_id=[[ENCOUNTERING_TASK:[0-f]+]]

/// CHECK: ompt_event_task_create
/// CHECK-SAME: new_task_id=[[TARGET_TASK_1:[0-f]+]]
/// CHECK-SAME: task_type=ompt_task_target

/// CHECK: ompt_event_task_create
/// CHECK-SAME: new_task_id=[[TARGET_TASK_2:[0-f]+]]
/// CHECK-SAME: task_type=ompt_task_target

/// CHECK: ompt_event_task_create
/// CHECK-SAME: new_task_id=[[TARGET_TASK_3:[0-f]+]]
/// CHECK-SAME: task_type=ompt_task_target

/// CHECK: ompt_event_task_create
/// CHECK-SAME: new_task_id=[[TARGET_TASK_4:[0-f]+]]
/// CHECK-SAME: task_type=ompt_task_target

/// CHECK: Callback Target EMI: kind=ompt_target_enter_data endpoint=ompt_scope_begin
/// CHECK-SAME: task_data=0x{{[0-f]+}} (0x[[ENCOUNTERING_TASK]])
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_1]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA:[0-f]+]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_1]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_1]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_1]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_1]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_1]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_1]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback Target EMI: kind=ompt_target_enter_data endpoint=ompt_scope_end
/// CHECK-SAME: task_data=0x{{[0-f]+}} (0x[[ENCOUNTERING_TASK]])
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_1]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_begin
/// CHECK-SAME: task_data=0x{{[0-f]+}} (0x[[ENCOUNTERING_TASK]])
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_2]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA:[0-f]+]])

/// CHECK: Callback Submit EMI: endpoint=ompt_scope_begin
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback Submit EMI: endpoint=ompt_scope_end
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_end
/// CHECK-SAME: task_data=0x{{[0-f]+}} (0x[[ENCOUNTERING_TASK]])
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_2]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_begin
/// CHECK-SAME: task_data=0x{{[0-f]+}} (0x[[ENCOUNTERING_TASK]])
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_3]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA:[0-f]+]])

/// CHECK: Callback Submit EMI: endpoint=ompt_scope_begin
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback Submit EMI: endpoint=ompt_scope_end
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback Target EMI: kind=ompt_target endpoint=ompt_scope_end
/// CHECK-SAME: task_data=0x{{[0-f]+}} (0x[[ENCOUNTERING_TASK]])
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_3]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback Target EMI: kind=ompt_target_exit_data endpoint=ompt_scope_begin
/// CHECK-SAME: task_data=0x{{[0-f]+}} (0x[[ENCOUNTERING_TASK]])
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_4]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA:[0-f]+]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_4]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_4]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_4]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_4]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_begin
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_4]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback DataOp EMI: endpoint=ompt_scope_end
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_4]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])

/// CHECK: Callback Target EMI: kind=ompt_target_exit_data endpoint=ompt_scope_end
/// CHECK-SAME: task_data=0x{{[0-f]+}} (0x[[ENCOUNTERING_TASK]])
/// CHECK-SAME: target_task_data=0x{{[0-f]+}} (0x[[TARGET_TASK_4]])
/// CHECK-SAME: target_data=0x{{[0-f]+}} (0x[[TARGET_DATA]])
