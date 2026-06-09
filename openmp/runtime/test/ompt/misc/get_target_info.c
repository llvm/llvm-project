// clang-format off
// RUN: %libomp-compile && %libomp-run | FileCheck %s
// REQUIRES: ompt
// clang-format on

/*
 * Example OpenMP program that checks if get_target_info entry point is
 * correctly implemented.
 */

#include <stdio.h>

#include <omp-tools.h>
#include <omp.h>

// From openmp/runtime/test/ompt/callback.h
#define register_ompt_callback_t(name, type)                                   \
  do {                                                                         \
    type f_##name = &on_##name;                                                \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_ompt_callback(name) register_ompt_callback_t(name, name##_t)

// OMPT entry point handles
static ompt_get_target_info_t ompt_get_target_info = 0;

// Init functions
int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_get_target_info = (ompt_get_target_info_t)lookup("ompt_get_target_info");
  if (!ompt_get_target_info)
    return 0; // failed

  return 1; // success
}

void ompt_finalize(ompt_data_t *tool_data) {}

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,
                                                            &ompt_finalize, 0};
  return &ompt_start_tool_result;
}

int main(void) {
  // Ensure OMPT is initialized
#pragma omp parallel
  {
  }

  uint64_t retrieved_device_num;
  ompt_id_t retrieved_target_id;
  ompt_id_t retrieved_host_op_id;
  int in_target_region = ompt_get_target_info(
      &retrieved_device_num, &retrieved_target_id, &retrieved_host_op_id);

  printf("omp_invalid_device: %lu\n", (uint64_t)omp_invalid_device);
  printf("ompt_id_none: %lu\n", ompt_id_none);
  printf("in target: %d | device_num: %lu | target_id: %lu | host_op_id: %lu\n",
         in_target_region, retrieved_device_num, retrieved_target_id,
         retrieved_host_op_id);
}

// clang-format off

/// CHECK: omp_invalid_device: [[OMP_INVALID_DEVICE:[0-9]+]]
/// CHECK: ompt_id_none: [[OMPT_ID_NONE:[0-9]+]]
/// CHECK: in target: 0
/// CHECK-SAME: device_num: [[OMP_INVALID_DEVICE]]
/// CHECK-SAME: target_id: [[OMPT_ID_NONE]]
/// CHECK-SAME: host_op_id: [[OMPT_ID_NONE]]

// clang-format on
