// RUN: %libomp-compile-and-run | FileCheck %s
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck --check-prefix=THREADS %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#include "callback.h"
#include <omp.h>
#include <unistd.h>

int main()
{
  omp_set_nested(1);
  int condition = 0;

  #pragma omp parallel num_threads(4)
  {
    print_ids(0);
    print_ids(1);
    //get all implicit task events before starting nested:
    #pragma omp barrier
    #pragma omp parallel num_threads(1)
    {
      print_ids(0);
      print_ids(1);
      print_ids(2);
      //get all implicit task events before starting nested:
      #pragma omp barrier
      #pragma omp parallel num_threads(4)
      {
        print_ids(0);
        print_ids(1);
        print_ids(2);
        print_ids(3);
      OMPT_SIGNAL(condition);
      OMPT_WAIT(condition,16);
      }
      print_fuzzy_address(1);
    }
    print_fuzzy_address(2);
  }
  print_fuzzy_address(3);

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquire'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquired'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_released'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // make sure initial data pointers are null
  // CHECK-NOT: 0: parallel_data initially not null
  // CHECK-NOT: 0: task_data initially not null
  // CHECK-NOT: 0: thread_data initially not null
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter={{0x[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra=0x{{[0-f]+}}, invoker=[[PARALLEL_INVOKER:[0-9]+]]

  // CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // Note that we cannot ensure that the worker threads have already called barrier_end and implicit_task_end before parallel_end!

  // CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]

  // CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]

  // CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]


  // THREADS: 0: NULL_POINTER=[[NULL:.*$]]
  // THREADS: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter=[[TASK_FRAME_ENTER:0x[0-f]+]], parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER:[0-9]+]]

  // nested parallel masters
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID:[0-9]+]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]

  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin: parent_task_id=[[IMPLICIT_TASK_ID]], parent_task_frame.exit=[[NESTED_TASK_FRAME_EXIT]], parent_task_frame.reenter=[[NESTED_TASK_FRAME_ENTER:0x[0-f]+]], parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=1, codeptr_ra=[[NESTED_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 2: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]

  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin: parent_task_id=[[NESTED_IMPLICIT_TASK_ID]], parent_task_frame.exit=[[NESTED_NESTED_TASK_FRAME_EXIT]], parent_task_frame.reenter=[[NESTED_NESTED_TASK_FRAME_ENTER:0x[0-f]+]], parallel_id=[[NESTED_NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra=[[NESTED_NESTED_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 2: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[MASTER_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[NESTED_NESTED_RETURN_ADDRESS]]
  // THREADS-NOT: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[NESTED_RETURN_ADDRESS]]
  // THREADS-NOT: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[PARALLEL_ID]], task_id=[[PARENT_TASK_ID]]
  // THREADS: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin: parent_task_id=[[IMPLICIT_TASK_ID]], parent_task_frame.exit=[[NESTED_TASK_FRAME_EXIT]], parent_task_frame.reenter=[[NESTED_TASK_FRAME_ENTER:0x[0-f]+]], parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=1, codeptr_ra=[[NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin: parent_task_id=[[NESTED_IMPLICIT_TASK_ID]], parent_task_frame.exit=[[NESTED_NESTED_TASK_FRAME_EXIT]], parent_task_frame.reenter=[[NESTED_NESTED_TASK_FRAME_ENTER:0x[0-f]+]], parallel_id=[[NESTED_NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra=[[NESTED_NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_end: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[NESTED_NESTED_RETURN_ADDRESS]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_end: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[NESTED_RETURN_ADDRESS]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin: parent_task_id=[[IMPLICIT_TASK_ID]], parent_task_frame.exit=[[NESTED_TASK_FRAME_EXIT]], parent_task_frame.reenter=[[NESTED_TASK_FRAME_ENTER:0x[0-f]+]], parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=1, codeptr_ra=[[NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin: parent_task_id=[[NESTED_IMPLICIT_TASK_ID]], parent_task_frame.exit=[[NESTED_NESTED_TASK_FRAME_EXIT]], parent_task_frame.reenter=[[NESTED_NESTED_TASK_FRAME_ENTER:0x[0-f]+]], parallel_id=[[NESTED_NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra=[[NESTED_NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_end: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[NESTED_NESTED_RETURN_ADDRESS]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_end: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[NESTED_RETURN_ADDRESS]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin: parent_task_id=[[IMPLICIT_TASK_ID]], parent_task_frame.exit=[[NESTED_TASK_FRAME_EXIT]], parent_task_frame.reenter=[[NESTED_TASK_FRAME_ENTER:0x[0-f]+]], parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=1, codeptr_ra=[[NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin: parent_task_id=[[NESTED_IMPLICIT_TASK_ID]], parent_task_frame.exit=[[NESTED_NESTED_TASK_FRAME_EXIT]], parent_task_frame.reenter=[[NESTED_NESTED_TASK_FRAME_ENTER:0x[0-f]+]], parallel_id=[[NESTED_NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra=[[NESTED_NESTED_RETURN_ADDRESS]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_NESTED_TASK_FRAME_EXIT:0x[0-f]+]], reenter_frame=[[NULL]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], exit_frame=[[NESTED_NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NESTED_TASK_FRAME_EXIT]], reenter_frame=[[NESTED_TASK_FRAME_ENTER]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_end: parallel_id=[[NESTED_NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[NESTED_NESTED_RETURN_ADDRESS]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[NESTED_IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_parallel_end: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // THREADS: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[NESTED_RETURN_ADDRESS]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // nested parallel worker threads
  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // THREADS: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // THREADS: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // can't reliably tell which parallel region is the parent...
  // THREADS: {{^}}[[THREAD_ID]]: task level 1: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 2: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}
  // THREADS: {{^}}[[THREAD_ID]]: task level 3: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[TASK_FRAME_ENTER]]
  // THREADS-NOT: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // THREADS: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // clang-format on

  return 0;
}
