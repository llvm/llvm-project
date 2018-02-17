// RUN: %libomp-cxx-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt

#include <iostream>
#include <thread>
#include "callback.h"
int condition = 0;
void f() {
  OMPT_SIGNAL(condition);
  // wait for both pthreads to arrive
  OMPT_WAIT(condition, 2);
  int i = 0;
#pragma omp parallel num_threads(2)
  {
    OMPT_SIGNAL(condition);
    OMPT_WAIT(condition, 6);
  }
}
int main() {
  std::thread t1(f);
  std::thread t2(f);
  t1.join();
  t2.join();
}

// Check if libomp supports the callbacks for this test.
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_schedule'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_end'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_thread_begin'

// CHECK: 0: NULL_POINTER=[[NULL:.*$]]

// first master thread
// CHECK: {{^}}[[MASTER_ID_1:[0-9]+]]: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_initial=1, thread_id=[[MASTER_ID_1]]

// CHECK: {{^}}[[MASTER_ID_1]]: ompt_event_task_create: parent_task_id=0
// CHECK-SAME: parent_task_frame.exit=[[NULL]]
// CHECK-SAME: parent_task_frame.reenter=[[NULL]]
// CHECK-SAME: new_task_id=[[PARENT_TASK_ID_1:[0-9]+]]
// CHECK-SAME: codeptr_ra=[[NULL]], task_type=ompt_task_initial=1
// CHECK-SAME: has_dependences=no

// CHECK: {{^}}[[MASTER_ID_1]]: ompt_event_parallel_begin:
// CHECK-SAME: parent_task_id=[[PARENT_TASK_ID_1]]
// CHECK-SAME: parent_task_frame.exit=[[NULL]]
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}}
// CHECK-SAME: parallel_id=[[PARALLEL_ID_1:[0-9]+]], requested_team_size=2
// CHECK-SAME: codeptr_ra=0x{{[0-f]+}}, invoker={{.*}}

// CHECK: {{^}}[[MASTER_ID_1]]: ompt_event_parallel_end:
// CHECK-SAME: parallel_id=[[PARALLEL_ID_1]], task_id=[[PARENT_TASK_ID_1]]
// CHECK-SAME: invoker={{[0-9]+}}

// CHECK: {{^}}[[MASTER_ID_1]]: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[MASTER_ID_1]]

// second master thread
// CHECK: {{^}}[[MASTER_ID_2:[0-9]+]]: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_initial=1, thread_id=[[MASTER_ID_2]]

// CHECK: {{^}}[[MASTER_ID_2]]: ompt_event_task_create: parent_task_id=0
// CHECK-SAME: parent_task_frame.exit=[[NULL]]
// CHECK-SAME: parent_task_frame.reenter=[[NULL]]
// CHECK-SAME: new_task_id=[[PARENT_TASK_ID_2:[0-9]+]]
// CHECK-SAME: codeptr_ra=[[NULL]], task_type=ompt_task_initial=1
// CHECK-SAME: has_dependences=no

// CHECK: {{^}}[[MASTER_ID_2]]: ompt_event_parallel_begin:
// CHECK-SAME: parent_task_id=[[PARENT_TASK_ID_2]]
// CHECK-SAME: parent_task_frame.exit=[[NULL]]
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}}
// CHECK-SAME: parallel_id=[[PARALLEL_ID_2:[0-9]+]]
// CHECK-SAME: requested_team_size=2, codeptr_ra=0x{{[0-f]+}}
// CHECK-SAME: invoker={{.*}}

// CHECK: {{^}}[[MASTER_ID_2]]: ompt_event_parallel_end:
// CHECK-SAME: parallel_id=[[PARALLEL_ID_2]], task_id=[[PARENT_TASK_ID_2]]
// CHECK-SAME: invoker={{[0-9]+}}

// CHECK: {{^}}[[MASTER_ID_2]]: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[MASTER_ID_2]]

// first worker thread
// CHECK: {{^}}[[THREAD_ID_1:[0-9]+]]: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID_1]]

// CHECK: {{^}}[[THREAD_ID_1]]: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[THREAD_ID_1]]

// second worker thread
// CHECK: {{^}}[[THREAD_ID_2:[0-9]+]]: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID_2]]

// CHECK: {{^}}[[THREAD_ID_2]]: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[THREAD_ID_2]]
