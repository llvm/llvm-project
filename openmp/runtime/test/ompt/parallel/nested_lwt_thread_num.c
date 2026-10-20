// clang-format off
// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// clang-format on

#include "callback.h"
#include <omp.h>

__attribute__ ((noinline)) // workaround for bug in icc
void print_task_info_at(int ancestor_level, int id)
{
#pragma omp critical
  {
    int task_type;
char buffer[2048];
ompt_data_t *parallel_data;
ompt_data_t *task_data;
int thread_num;
ompt_get_task_info(ancestor_level, &task_type, &task_data, NULL, &parallel_data,
                   &thread_num);
format_task_type(task_type, buffer);
printf("%" PRIu64 ": ancestor_level=%d id=%d task_type=%s=%d "
       "parallel_id=%" PRIx64 " task_id=%" PRIx64 " thread_num=%d\n",
       ompt_get_thread_data()->value, ancestor_level, id, buffer, task_type,
       parallel_data->value, task_data->value, thread_num);
}
}
;

__attribute__ ((noinline)) // workaround for bug in icc
void print_innermost_task_info(int id)
{
  print_task_info_at(0, id);
}

int main() {

#pragma omp parallel num_threads(2)
  {
    // sync threads before checking the output
#pragma omp barrier
    // region 0
    if (omp_get_thread_num() == 1) {
      // executed by worker thread only
      // assert that thread_num is 1
      print_innermost_task_info(1);

#pragma omp parallel num_threads(1)
      {
        // serialized region 1
        // assert that thread_num is 0
        print_innermost_task_info(2);

#pragma omp parallel num_threads(1)
        {
          // serialized region 2
          // assert that thread_num is 0
          print_innermost_task_info(3);

          // Check the value of thread_num while iterating over the hierarchy
          // of active tasks.
          print_task_info_at(0, 3);
          print_task_info_at(1, 2);
          print_task_info_at(2, 1);
        }
      }
    }
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_initial_task_begin: parallel_id=[[PARALLEL_ID_0:[0-9]+]], task_id=[[TASK_ID_0:[0-9]+]], actual_parallelism=1, index=1, flags=1

  // region 0
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin: parent_task_id=[[TASK_ID_0]],
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_1:[0-9]+]]
  // CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID_1]], task_id=[[TASK_ID_1:[0-9]+]]
  // CHECK-DAG: {{^}}[[WORKER_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID_1]], task_id=[[TASK_ID_2:[0-9]+]]
  // assert some info about implicit task executed by worker thread
  // thread_num is the most important
  // CHECK: {{^}}[[WORKER_ID]]: ancestor_level=0 id=1
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_1]] task_id=[[TASK_ID_2]]
  // CHECK-SAME: thread_num=1

  // serialized region 1
  // CHECK: {{^}}[[WORKER_ID]]: ompt_event_parallel_begin: parent_task_id=[[TASK_ID_2]],
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_2:[0-9]+]]
  // CHECK-DAG: {{^}}[[WORKER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID_2]], task_id=[[TASK_ID_3:[0-9]+]]
  // assert some information about the implicit task of the serialized region 1
  // pay attention that thread_num should take value 0
  // CHECK: {{^}}[[WORKER_ID]]: ancestor_level=0 id=2
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_2]] task_id=[[TASK_ID_3]]
  // CHECK-SAME: thread_num=0

  // serialized region 2
  // CHECK: {{^}}[[WORKER_ID]]: ompt_event_parallel_begin: parent_task_id=[[TASK_ID_3]],
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_3:[0-9]+]]
  // CHECK-DAG: {{^}}[[WORKER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID_3]], task_id=[[TASK_ID_4:[0-9]+]]
  // assert some information about the implicit task of the serialized region 2
  // pay attention that thread_num should take value 0
  // CHECK: {{^}}[[WORKER_ID]]: ancestor_level=0 id=3
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_3]] task_id=[[TASK_ID_4]]
  // CHECK-SAME: thread_num=0

  // Check the value of thread_num argument while iterating over the hierarchy
  // of active tasks. The expected is that thread_num takes the value checked
  // above in the test case (0, 0, 1 - respectively).

  // Thread is the master thread of the region 2, so thread_num should be 0.
  // CHECK: {{^}}[[WORKER_ID]]: ancestor_level=0 id=3
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_3]] task_id=[[TASK_ID_4]]
  // CHECK-SAME: thread_num=0

  // Thread is the master thread of the region 1, so thread_num should be 0.
  // CHECK: {{^}}[[WORKER_ID]]: ancestor_level=1 id=2
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_2]] task_id=[[TASK_ID_3]]
  // CHECK-SAME: thread_num=0

  // Thread is the worker thread of the region 0, so thread_num should be 1.
  // CHECK: {{^}}[[WORKER_ID]]: ancestor_level=2 id=1
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_1]] task_id=[[TASK_ID_2]]
  // CHECK-SAME: thread_num=1
  // clang-format on

  return 0;
}
