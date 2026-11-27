// clang-format off
// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// clang-format on

#define USE_PRIVATE_TOOL 1
#include "callback.h"

__attribute__((noinline)) int foo(int x) {
#pragma omp parallel num_threads(2)
  {
#pragma omp atomic
    x++;
  }
  return x;
}

__attribute__((noinline)) int bar(int x) {
#pragma omp parallel num_threads(2)
  {
#pragma omp critical
    x++;
  }
  return x;
}

int main() {
  int y;
  y = foo(y);
  y = bar(y);
  y = foo(y);
  return 0;

  // clang-format off
  // CHECK-NOT: {{^}}0: Could not register callback
  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // First call to foo
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin
  // CHECK-SAME: {{.*}}codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]

  // Call to bar
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin

  // Second call to foo
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin
  // CHECK-SAME: {{.*}}codeptr_ra=[[RETURN_ADDRESS]]
  // clang-format on
}

static void on_ompt_callback_thread_begin(ompt_thread_t thread_type,
                                          ompt_data_t *thread_data) {
  if (thread_data->ptr)
    printf("%s\n", "0: thread_data initially not null");
  thread_data->value = ompt_get_unique_id();
  printf("%" PRIu64 ":" _TOOL_PREFIX
         " ompt_event_thread_begin: thread_type=%s=%d, thread_id=%" PRIu64 "\n",
         ompt_get_thread_data()->value, ompt_thread_t_values[thread_type],
         thread_type, thread_data->value);
}

static void on_ompt_callback_parallel_begin(
    ompt_data_t *encountering_task_data,
    const ompt_frame_t *encountering_task_frame, ompt_data_t *parallel_data,
    uint32_t requested_team_size, int flag, const void *codeptr_ra) {
  if (parallel_data->ptr)
    printf("0: parallel_data initially not null\n");
  parallel_data->value = ompt_get_unique_id();
  int invoker = flag & 0xF;
  const char *event = (flag & ompt_parallel_team) ? "parallel" : "teams";
  const char *size = (flag & ompt_parallel_team) ? "team_size" : "num_teams";
  printf("%" PRIu64 ":" _TOOL_PREFIX
         " ompt_event_%s_begin: parent_task_id=%" PRIu64
         ", parent_task_frame.exit=%p, parent_task_frame.reenter=%p, "
         "parallel_id=%" PRIu64 ", requested_%s=%" PRIu32
         ", codeptr_ra=%p, invoker=%d\n",
         ompt_get_thread_data()->value, event, encountering_task_data->value,
         encountering_task_frame->exit_frame.ptr,
         encountering_task_frame->enter_frame.ptr, parallel_data->value, size,
         requested_team_size, codeptr_ra, invoker);
}

int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
  ompt_get_unique_id = (ompt_get_unique_id_t)lookup("ompt_get_unique_id");
  ompt_get_thread_data = (ompt_get_thread_data_t)lookup("ompt_get_thread_data");

  register_ompt_callback(ompt_callback_thread_begin);
  register_ompt_callback(ompt_callback_parallel_begin);
  printf("0: NULL_POINTER=%p\n", (void *)NULL);
  return 1; // success
}

void ompt_finalize(ompt_data_t *tool_data) {}

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,
                                                            &ompt_finalize, 0};
  return &ompt_start_tool_result;
}
