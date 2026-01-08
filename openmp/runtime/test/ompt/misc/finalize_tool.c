// clang-format off
// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// clang-format on
#include "callback.h"
#include "omp_testsuite.h"

int main() {
  go_parallel_nthreads(2);

  printf("Before ompt_finalize_tool\n");
  ompt_finalize_tool();
  printf("After ompt_finalize_tool\n");

  return get_exit_value();
}

// clang-format off
// CHECK: 0: NULL_POINTER=[[NULL:.*$]]
// CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_initial=1

// CHECK: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin
// CHECK: {{^}}[[THREAD_ID]]: ompt_event_parallel_end

// CHECK: {{^}}Before ompt_finalize_tool

// CHECK: {{^}}[[THREAD_ID]]: ompt_event_thread_end: thread_id=[[THREAD_ID]]
// CHECK: 0: ompt_event_runtime_shutdown

// CHECK: {{^}}After ompt_finalize_tool
// clang-format on
