// clang-format off
// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
// clang-format on
#define TEST_NEED_PRINT_FRAME_FROM_OUTLINED_FN
#include "callback.h"
#include <omp.h>
#include <stdio.h>

int main() {
  int result = omp_control_tool(omp_control_tool_flush, 1, NULL);
  printf("control_tool result = %d\n", result);

#pragma omp parallel num_threads(1)
  {
    print_frame_from_outlined_fn(1);
    print_frame(0);
    print_current_address(0);
  }

  result = omp_control_tool(omp_control_tool_flush, 1, NULL);
  printf("control_tool result = %d\n", result);

  // clang-format off

  // Check if libomp allows interacting with an attached tool both before and after the first
  // directive is being used.

  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_control_tool'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_control_tool: command=3, modifier=1, arg=[[NULL]], codeptr_ra={{(0x)?[0-f]*}}, current_task_frame.exit={{.*}}, current_task_frame.reenter={{(0x)?[0-f]*}}
  // CHECK-NEXT: control_tool result = 0
  
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_control_tool: command=3, modifier=1, arg=[[NULL]], codeptr_ra={{(0x)?[0-f]*}}, current_task_frame.exit={{.*}}, current_task_frame.reenter={{(0x)?[0-f]*}}
  // CHECK-NEXT: control_tool result = 0

  // clang-format on

  return 0;
}
