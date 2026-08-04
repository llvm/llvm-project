// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt

#include <omp-tools.h>
#include <stdio.h>

void ompt_libomp_connect(ompt_start_tool_result_t *result);

static int tool_initialize(ompt_function_lookup_t lookup,
                           int initial_device_num, ompt_data_t *tool_data) {
  return 1;
}

static int target_initialize(ompt_function_lookup_t lookup,
                             int initial_device_num, ompt_data_t *tool_data) {
  printf("ompt_set_frame_enter: %s\n",
         lookup("ompt_set_frame_enter") ? "present" : "absent");
  return 1;
}

static void ompt_finalize(ompt_data_t *tool_data) {}

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t result = {&tool_initialize, &ompt_finalize,
                                            0};
  return &result;
}

int main() {
  static ompt_start_tool_result_t result = {&target_initialize, &ompt_finalize,
                                            0};
  // This entry point initializes OMPT before invoking the target connector.
  ompt_libomp_connect(&result);
  return 0;
}

// CHECK: ompt_set_frame_enter: absent
