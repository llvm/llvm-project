// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt

/// GCC lowering of distribute results in calls to 
/// omp_get_num_teams/omp_get_team_num rather than region calls
// UNSUPPORTED: gcc
#include "callback.h"

#define WORK_SIZE 64

int main() {
  int i;
#pragma omp teams num_teams(4) thread_limit(1)
#pragma omp distribute dist_schedule(static, WORK_SIZE / 4)
  for (i = 0; i < WORK_SIZE; i++) {}

  return 0;
}

// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_work'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_dispatch'

// CHECK: 0: NULL_POINTER=[[NULL:.*$]]

// CHECK: {{^}}[[THREAD_ID0:[0-9]+]]: ompt_event_distribute_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID0:[0-9]+]]
// CHECK-SAME: parent_task_id=[[TASK_ID0:[0-9]+]]
// CHECK: {{^}}[[THREAD_ID0]]: ompt_event_distribute_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID0]], task_id=[[TASK_ID0]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=16

// CHECK: {{^}}[[THREAD_ID1:[0-9]+]]: ompt_event_distribute_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID1:[0-9]+]]
// CHECK-SAME: parent_task_id=[[TASK_ID1:[0-9]+]]
// CHECK: {{^}}[[THREAD_ID1]]: ompt_event_distribute_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID1]], task_id=[[TASK_ID1]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=16

// CHECK: {{^}}[[THREAD_ID2:[0-9]+]]: ompt_event_distribute_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID2:[0-9]+]]
// CHECK-SAME: parent_task_id=[[TASK_ID2:[0-9]+]]
// CHECK: {{^}}[[THREAD_ID2]]: ompt_event_distribute_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID2]], task_id=[[TASK_ID2]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=16

// CHECK: {{^}}[[THREAD_ID3:[0-9]+]]: ompt_event_distribute_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID3:[0-9]+]]
// CHECK-SAME: parent_task_id=[[TASK_ID3:[0-9]+]]
// CHECK: {{^}}[[THREAD_ID3]]: ompt_event_distribute_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID3]], task_id=[[TASK_ID3]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=16
