// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main() {

  int condition = 0;
  int x = 0;
  omp_set_nested(1);
#pragma omp parallel num_threads(2)
  {
#pragma omp parallel num_threads(2)
    {
      OMPT_SIGNAL(condition);
      OMPT_WAIT(condition, 4);
    }
  }

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_thread_begin:
  // CHECK-SAME: thread_type=ompt_thread_initial=1, thread_id=[[MASTER_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_thread_end:
  // CHECK-SAME: thread_id=[[MASTER_ID]]
  // CHECK: {{^}}[[WORKER_ID1:[0-9]+]]: ompt_event_thread_begin:
  // CHECK-SAME: thread_type=ompt_thread_worker=2, thread_id=[[WORKER_ID1]]
  // CHECK: {{^}}[[WORKER_ID1]]: ompt_event_thread_end:
  // CHECK-SAME: thread_id=[[WORKER_ID1]]
  // CHECK: {{^}}[[WORKER_ID2:[0-9]+]]: ompt_event_thread_begin:
  // CHECK-SAME: thread_type=ompt_thread_worker=2, thread_id=[[WORKER_ID2]]
  // CHECK: {{^}}[[WORKER_ID2]]: ompt_event_thread_end:
  // CHECK-SAME: thread_id=[[WORKER_ID2]]
  // CHECK: {{^}}[[WORKER_ID3:[0-9]+]]: ompt_event_thread_begin:
  // CHECK-SAME: thread_type=ompt_thread_worker=2, thread_id=[[WORKER_ID3]]
  // CHECK: {{^}}[[WORKER_ID3]]: ompt_event_thread_end:
  // CHECK-SAME: thread_id=[[WORKER_ID3]]

  return 0;
}
