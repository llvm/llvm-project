// clang-format off
// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
// clang-format on
#define TEST_NEED_PRINT_FRAME_FROM_OUTLINED_FN
#define _OMPT_DISABLE_CONTROL_TOOL
#include "callback.h"
#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel num_threads(1)
  {
    int result = omp_control_tool(omp_control_tool_flush, 1, NULL);
    printf("control_tool result = %d\n", result);
  }

  // clang-format off
  // Check if libomp returns -1 (no callback) when a tool is attached,
  // but ompt_callback_control_tool is not registered

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: control_tool result = -1

  // clang-format on

  return 0;
}
