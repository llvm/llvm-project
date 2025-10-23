#define SKIP_CALLBACK_REGISTRATION 1

#include "../../../openmp/runtime/test/ompt/callback.h"
#include "callbacks.h"
#include <omp-tools.h>

// From openmp/runtime/test/ompt/callback.h
#define register_ompt_callback_t(name, type)                                   \
  do {                                                                         \
    type f_##name = &on_##name;                                                \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_ompt_callback(name) register_ompt_callback_t(name, name##_t)

// Init functions
int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");

  if (!ompt_set_callback)
    return 0; // failed

  // host runtime functions
  ompt_get_unique_id = (ompt_get_unique_id_t)lookup("ompt_get_unique_id");
  ompt_get_thread_data = (ompt_get_thread_data_t)lookup("ompt_get_thread_data");
  ompt_get_task_info = (ompt_get_task_info_t)lookup("ompt_get_task_info");

  ompt_get_unique_id();

  // host callbacks
  register_ompt_callback(ompt_callback_sync_region);
  register_ompt_callback_t(ompt_callback_sync_region_wait,
                           ompt_callback_sync_region_t);
  register_ompt_callback_t(ompt_callback_reduction,
                           ompt_callback_sync_region_t);
  register_ompt_callback(ompt_callback_implicit_task);
  register_ompt_callback(ompt_callback_parallel_begin);
  register_ompt_callback(ompt_callback_parallel_end);
  register_ompt_callback(ompt_callback_task_create);
  register_ompt_callback(ompt_callback_task_schedule);

  // device callbacks
  register_ompt_callback(ompt_callback_device_initialize);
  register_ompt_callback(ompt_callback_device_finalize);
  register_ompt_callback(ompt_callback_device_load);
  register_ompt_callback(ompt_callback_target_data_op_emi);
  register_ompt_callback(ompt_callback_target_emi);
  register_ompt_callback(ompt_callback_target_submit_emi);

  return 1; // success
}

void ompt_finalize(ompt_data_t *tool_data) {}

#ifdef __cplusplus
extern "C" {
#endif
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,
                                                            &ompt_finalize, 0};
  return &ompt_start_tool_result;
}
#ifdef __cplusplus
}
#endif
