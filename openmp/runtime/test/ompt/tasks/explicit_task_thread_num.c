// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt

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
    ompt_get_task_info(ancestor_level, &task_type, &task_data, NULL,
                       &parallel_data, &thread_num);
    format_task_type(task_type, buffer);
    printf("%" PRIu64 ": ancestor_level=%d id=%d task_type=%s=%d "
                      "parallel_id=%" PRIu64 " task_id=%" PRIu64
                      " thread_num=%d\n",
        ompt_get_thread_data()->value, ancestor_level, id, buffer,
        task_type, parallel_data->value, task_data->value, thread_num);
  }
};

int main()
{

#pragma omp parallel num_threads(2)
  {

    if (omp_get_thread_num() == 1) {
      // To assert that task is executed by the worker thread,
      // if(0) is used in order to ensure that the task is immediately
      // executed after its creation.
#pragma omp task if(0)
      {
        // thread_num should be equal to 1 for both explicit and implicit task
        print_task_info_at(0, 1);
        print_task_info_at(1, 0);
      };
    }
  }

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_parallel_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'

  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_initial_task_begin

  // parallel region used only to determine worker thread id
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin
  // CHECK: {{^}}[[WID:[0-9]+]]: ompt_event_implicit_task{{.*}}thread_num=1

  // thread_num must be equal to 1 for both explicit and the implicit tasks
  // CHECK: {{^}}[[WID]]: ancestor_level=0 id=1 task_type=ompt_task_explicit
  // CHECK-SAME: thread_num=1
  // CHECK: {{^}}[[WID]]: ancestor_level=1 id=0 task_type=ompt_task_implicit
  // CHECK-SAME: thread_num=1

  return 0;
}
