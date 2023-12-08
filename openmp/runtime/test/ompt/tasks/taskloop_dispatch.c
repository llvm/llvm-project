// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gnu, intel-16.0

#include "callback.h"
#include <omp.h>

int main() {
  unsigned int i, x;

#pragma omp parallel num_threads(2)
  {
#pragma omp barrier

#pragma omp master
#pragma omp taskloop grainsize(4)
    for (i = 0; i < 16; i++) {
      // Make every iteration takes at least 1ms
      delay(1000);
    }
  }
  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
  // CHECK-SAME: parent_task_id={{[0-9]+}}
  // CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]]
  // CHECK-SAME: requested_team_size=2
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: task_id=[[IMPLICIT_TASK_ID1:[0-9]+]]
  // CHECK-SAME: team_size=2, thread_num=0

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskloop_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: parent_task_id=[[IMPLICIT_TASK_ID1]]
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]], count=16

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: new_task_id=[[TASK_ID0:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: new_task_id=[[TASK_ID1:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: new_task_id=[[TASK_ID2:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: new_task_id=[[TASK_ID3:[0-9]+]]

  // CHECK-DAG: {{.*}}: ompt_event_taskloop_chunk_begin:{{.*}}task_id=[[TASK_ID0]]{{.*}}chunk_iterations=4
  // CHECK-DAG: {{.*}}: ompt_event_taskloop_chunk_begin:{{.*}}task_id=[[TASK_ID1]]{{.*}}chunk_iterations=4
  // CHECK-DAG: {{.*}}: ompt_event_taskloop_chunk_begin:{{.*}}task_id=[[TASK_ID2]]{{.*}}chunk_iterations=4
  // CHECK-DAG: {{.*}}: ompt_event_taskloop_chunk_begin:{{.*}}task_id=[[TASK_ID3]]{{.*}}chunk_iterations=4

  return 0;
}
