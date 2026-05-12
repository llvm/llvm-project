// clang-format off
// RUN: %libomp-compile-and-run
// clang-format on

#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel num_threads(1)
  {
    int result = omp_control_tool(omp_control_tool_flush, 1, NULL);
    printf("control_tool result = %d\n", result);
  }

  // clang-format off
  // Check if libomp correctly reports -2 (no tool) if no tool is attached.

  // CHECK: control_tool result = -2
  // clang-format on

  return 0;
}
