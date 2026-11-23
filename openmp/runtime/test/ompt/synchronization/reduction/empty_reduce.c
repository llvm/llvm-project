// clang-format off
// RUN: %libomp-compile-and-run | FileCheck %s
// RUN: %libomp-compile -DNOWAIT && %libomp-run | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc
// clang-format on
#include "callback.h"
#include <omp.h>

#ifdef NOWAIT
#define FOR_CLAUSE nowait
#else
#define FOR_CLAUSE
#endif

int main() {
  int sum = 0;
  int i;
#pragma omp parallel num_threads(1)
#pragma omp for reduction(+ : sum) FOR_CLAUSE
  for (i = 0; i < 10000; i++) {
    sum += i;
  }

  // clang-format off
  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID:[0-f]+]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_reduction_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]],
  // CHECK-SAME: codeptr_ra=
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_reduction_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]],
  // CHECK-SAME: task_id=[[TASK_ID]], codeptr_ra=
  // clang-format on

  return 0;
}
