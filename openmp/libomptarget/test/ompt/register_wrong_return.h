#include <omp-tools.h>

// From openmp/runtime/test/ompt/callback.h
#define register_ompt_callback_t(name, type)                                   \
  do {                                                                         \
    type f_##name = &on_##name;                                                \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_ompt_callback(name) register_ompt_callback_t(name, name##_t)

// OMPT entry point handles
static ompt_set_callback_t ompt_set_callback = 0;

// Init functions
int ompt_initialize(
  ompt_function_lookup_t lookup,
  int initial_device_num,
  ompt_data_t *tool_data)
{
  ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");

  if (!ompt_set_callback) return 1; // failed but wrongly returning 1
  
  register_ompt_callback(ompt_callback_device_initialize);
  register_ompt_callback(ompt_callback_device_finalize);
  register_ompt_callback(ompt_callback_device_load);
  register_ompt_callback(ompt_callback_target_data_op);
  register_ompt_callback(ompt_callback_target);
  register_ompt_callback(ompt_callback_target_submit);

  return 0; //success but should return 1 according to the spec
}

void ompt_finalize(ompt_data_t *tool_data)
{
}

#ifdef __cplusplus
extern "C" {
#endif
ompt_start_tool_result_t *ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,&ompt_finalize, 0};
  return &ompt_start_tool_result;
}
#ifdef __cplusplus
}
#endif
