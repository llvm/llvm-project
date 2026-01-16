// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// RUN: %libomp-compile -DNOWAIT && %libomp-run | %sort-threads | FileCheck %s
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
  int sum = 0, a = 0, b = 0;
  int i;
#pragma omp parallel num_threads(5)
// for 32-bit architecture we need at least 3 variables to trigger tree
#pragma omp for reduction(+ : sum, a, b) FOR_CLAUSE
  for (i = 0; i < 10000; i++) {
    a = b = sum += i;
  }

  printf("%i\n", sum);
  // clang-format off
  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID:[0-f]+]]

  // order and distribution to threads not determined
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-f]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-f]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-f]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-f]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-f]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-f]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-f]+}}
  // CHECK: {{^}}{{[0-f]+}}: ompt_event_reduction_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-f]+}}
  // clang-format on

  return 0;
}
