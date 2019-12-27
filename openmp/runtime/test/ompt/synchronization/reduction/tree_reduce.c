// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc
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
#pragma omp parallel num_threads(5)
#pragma omp for reduction(+ : sum) FOR_CLAUSE
  for (i = 0; i < 10000; i++) {
    sum += i;
  }

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID:[0-9]+]]

  // order and distribution to threads not determined
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}

  return 0;
}
