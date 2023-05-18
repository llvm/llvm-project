// RUN: %libomp-cxx-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt

#include <iostream>
#include <thread>
#if !defined(__FreeBSD__) && !defined(__NetBSD__)
#include <alloca.h>
#else
#include <cstdlib>
#endif

#include "callback.h"
#include "omp.h"

int condition = 0;

void f() {
  // Call OpenMP API function to force initialization of OMPT.
  // (omp_get_thread_num() does not work because it just returns 0 if the
  // runtime isn't initialized yet...)
  omp_get_num_threads();

  // Call alloca() to force availability of frame pointer
  void *p = alloca(0);

  OMPT_SIGNAL(condition);
  // Wait for both initial threads to arrive that will eventually become the
  // master threads in the following parallel region.
  OMPT_WAIT(condition, 2);

#pragma omp parallel num_threads(2)
  {
    // Wait for all threads to arrive so that no worker thread can be reused...
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
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_schedule'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_end'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_thread_begin'

// CHECK: 0: NULL_POINTER=[[NULL:.*$]]

// first master thread
// CHECK: {{^}}[[MASTER_ID_1:[0-9]+]]: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_initial=1, thread_id=[[MASTER_ID_1]]


// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_initial_task_begin: parallel_id={{[0-9]+}}
// CHECK-SAME: task_id=[[PARENT_TASK_ID_1:[0-9]+]], actual_parallelism=1,
// CHECK-SAME: index=1, flags=1

// CHECK: {{^}}[[MASTER_ID_1]]: ompt_event_parallel_begin:
// CHECK-SAME: parent_task_id=[[PARENT_TASK_ID_1]]
// CHECK-SAME: parent_task_frame.exit=[[NULL]]
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}}
// CHECK-SAME: parallel_id=[[PARALLEL_ID_1:[0-9]+]], requested_team_size=2
// CHECK-SAME: codeptr_ra=0x{{[0-f]+}}, invoker={{.*}}

// CHECK: {{^}}[[MASTER_ID_1]]: ompt_event_parallel_end:
// CHECK-SAME: parallel_id=[[PARALLEL_ID_1]], task_id=[[PARENT_TASK_ID_1]]
// CHECK-SAME: invoker={{[0-9]+}}

// CHECK: {{^}}[[MASTER_ID_1]]: ompt_event_initial_task_end:
// CHECK-SAME: parallel_id={{[0-9]+}}, task_id=[[PARENT_TASK_ID_1]],
// CHECK-SAME: actual_parallelism=0, index=1

// CHECK: {{^}}[[MASTER_ID_1]]: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[MASTER_ID_1]]

// second master thread
// CHECK: {{^}}[[MASTER_ID_2:[0-9]+]]: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_initial=1, thread_id=[[MASTER_ID_2]]

// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_initial_task_begin: parallel_id={{[0-9]+}}
// CHECK-SAME: task_id=[[PARENT_TASK_ID_2:[0-9]+]], actual_parallelism=1,
// CHECK-SAME: index=1, flags=1

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

// CHECK: {{^}}[[MASTER_ID_2]]: ompt_event_initial_task_end:
// CHECK-SAME: parallel_id={{[0-9]+}}, task_id=[[PARENT_TASK_ID_2]],
// CHECK-SAME: actual_parallelism=0, index=1

// CHECK: {{^}}[[MASTER_ID_2]]: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[MASTER_ID_2]]

// first worker thread
// CHECK: {{^}}[[THREAD_ID_1:[0-9]+]]: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID_1]]
// CHECK-NOT: {{^}}[[THREAD_ID_1:[0-9]+]]: ompt_event_initial_task_end:

// CHECK: {{^}}[[THREAD_ID_1]]: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[THREAD_ID_1]]

// second worker thread
// CHECK: {{^}}[[THREAD_ID_2:[0-9]+]]: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID_2]]

// CHECK: {{^}}[[THREAD_ID_2]]: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[THREAD_ID_2]]
