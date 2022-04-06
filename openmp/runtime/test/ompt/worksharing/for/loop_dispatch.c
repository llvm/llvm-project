// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// XFAIL: gcc
// GCC doesn't call runtime for static schedule

#include "callback.h"

#define WORK_SIZE 64

int main() {
  int i;

#pragma omp parallel num_threads(4)
  {
#pragma omp for schedule(static, WORK_SIZE / 4)
    for (i = 0; i < WORK_SIZE; i++) {}

#pragma omp for schedule(dynamic)
    for (i = 0; i < WORK_SIZE; i++) {
      // Make every iteration takes at least 1ms.
      delay(1000);
    }

#pragma omp for schedule(guided)
    for (i = 0; i < WORK_SIZE; i++) {
      // Make every iteration takes at least 1ms.
      delay(1000);
    }
  }

  return 0;
}

// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_work'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_dispatch'

// CHECK: 0: NULL_POINTER=[[NULL:.*$]]
// CHECK: {{^}}[[THREAD_ID0:[0-9]+]]: ompt_event_parallel_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]]

// Each thread should have at least one ws-loop-chunk-begin event for each
// for loop.

// CHECK: {{^}}[[THREAD_ID0]]: ompt_event_implicit_task_begin:
// CHECK-SAME: task_id=[[TASK_ID0:[0-9]+]]
// CHECK: {{^}}[[THREAD_ID0]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID0]]
// CHECK: {{^}}[[THREAD_ID0]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID0]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=16
// CHECK: {{^}}[[THREAD_ID0]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID0]]
// CHECK: {{^}}[[THREAD_ID0]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID0]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=1
// CHECK: {{^}}[[THREAD_ID0]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID0]]
// CHECK: {{^}}[[THREAD_ID0]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID0]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations={{[1-9][0-9]*}}

// CHECK: {{^}}[[THREAD_ID1:[0-9]+]]: ompt_event_implicit_task_begin:
// CHECK-SAME: task_id=[[TASK_ID1:[0-9]+]]
// CHECK: {{^}}[[THREAD_ID1]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID1]]
// CHECK: {{^}}[[THREAD_ID1]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID1]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=16
// CHECK: {{^}}[[THREAD_ID1]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID1]]
// CHECK: {{^}}[[THREAD_ID1]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID1]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=1
// CHECK: {{^}}[[THREAD_ID1]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID1]]
// CHECK: {{^}}[[THREAD_ID1]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID1]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations={{[1-9][0-9]*}}

// CHECK: {{^}}[[THREAD_ID2:[0-9]+]]: ompt_event_implicit_task_begin:
// CHECK-SAME: task_id=[[TASK_ID2:[0-9]+]]
// CHECK: {{^}}[[THREAD_ID2]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID2]]
// CHECK: {{^}}[[THREAD_ID2]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID2]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=16
// CHECK: {{^}}[[THREAD_ID2]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID2]]
// CHECK: {{^}}[[THREAD_ID2]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID2]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=1
// CHECK: {{^}}[[THREAD_ID2]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID2]]
// CHECK: {{^}}[[THREAD_ID2]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID2]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations={{[1-9][0-9]*}}

// CHECK: {{^}}[[THREAD_ID3:[0-9]+]]: ompt_event_implicit_task_begin:
// CHECK-SAME: task_id=[[TASK_ID3:[0-9]+]]
// CHECK: {{^}}[[THREAD_ID3]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID3]]
// CHECK: {{^}}[[THREAD_ID3]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID3]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=16
// CHECK: {{^}}[[THREAD_ID3]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID3]]
// CHECK: {{^}}[[THREAD_ID3]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID3]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations=1
// CHECK: {{^}}[[THREAD_ID3]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID3]]
// CHECK: {{^}}[[THREAD_ID3]]: ompt_event_ws_loop_chunk_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID3]]
// CHECK-SAME: chunk_start={{[0-9]+}}, chunk_iterations={{[1-9][0-9]*}}
