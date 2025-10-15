// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// clang-format on
#include <omp.h>
#include "callback.h"

int main() {
  omp_set_nested(1);
#pragma omp parallel num_threads(2)
  {
#pragma omp barrier
#pragma omp parallel num_threads(2)
    { print_frame(0); }
  }

  // clang-format off
  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: thread_num=[[OUTER_THREAD_NUM1:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin:
  // CHECK-SAME: parallel_id=[[INNER_PARALLEL_ID1:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: parallel_id=[[INNER_PARALLEL_ID1]]
  // CHECK-SAME: thread_num=[[INNER_THREAD_NUM1:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
  // CHECK-SAME: thread_num=[[INNER_THREAD_NUM1]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end:
  // CHECK-SAME: parallel_id=[[INNER_PARALLEL_ID1]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
  // CHECK-SAME: thread_num=[[OUTER_THREAD_NUM1]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]

  // CHECK: {{^}}[[WORKER_ID1:[0-9]+]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: thread_num=[[OUTER_THREAD_NUM2:[0-9]+]]
  // CHECK: {{^}}[[WORKER_ID1]]: ompt_event_parallel_begin:
  // CHECK-SAME: parallel_id=[[INNER_PARALLEL_ID2:[0-9]+]]
  // CHECK: {{^}}[[WORKER_ID1]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: parallel_id=[[INNER_PARALLEL_ID2]]
  // CHECK-SAME: thread_num=[[INNER_THREAD_NUM2:[0-9]+]]
  // CHECK: {{^}}[[WORKER_ID1]]: ompt_event_implicit_task_end
  // CHECK-SAME: thread_num=[[INNER_THREAD_NUM2]]
  // CHECK: {{^}}[[WORKER_ID1]]: ompt_event_parallel_end:
  // CHECK-SAME: parallel_id=[[INNER_PARALLEL_ID2]]
  // CHECK: {{^}}[[WORKER_ID1]]: ompt_event_implicit_task_end
  // CHECK-SAME: thread_num=[[OUTER_THREAD_NUM2]]

  // CHECK: {{^}}[[WORKER_ID2:[0-9]+]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: thread_num=[[INNER_THREAD_NUM3:[0-9]+]]
  // CHECK: {{^}}[[WORKER_ID2]]: ompt_event_implicit_task_end
  // CHECK-SAME: thread_num=[[INNER_THREAD_NUM3]]

  // CHECK: {{^}}[[WORKER_ID3:[0-9]+]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: thread_num=[[INNER_THREAD_NUM4:[0-9]+]]
  // CHECK: {{^}}[[WORKER_ID3]]: ompt_event_implicit_task_end
  // CHECK-SAME: thread_num=[[INNER_THREAD_NUM4]]
  // clang-format on

  return 0;
}
