// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// GCC generates code that does not call the runtime for the master construct
// XFAIL: gcc

#define USE_PRIVATE_TOOL 1
#include "callback.h"
#include <omp.h>

int main() {
  int x = 0;
#pragma omp parallel num_threads(2)
  {
#pragma omp master
    {
      print_fuzzy_address(1);
      x++;
    }
    print_current_address(2);
  }

  printf("%" PRIu64 ": x=%d\n", ompt_get_thread_data()->value, x);

  return 0;
}

static void on_ompt_callback_master(ompt_scope_endpoint_t endpoint,
                                    ompt_data_t *parallel_data,
                                    ompt_data_t *task_data,
                                    const void *codeptr_ra) {
  switch (endpoint) {
  case ompt_scope_begin:
    printf("%" PRIu64 ":" _TOOL_PREFIX
           " ompt_event_master_begin: codeptr_ra=%p\n",
           ompt_get_thread_data()->value, codeptr_ra);
    break;
  case ompt_scope_end:
    printf("%" PRIu64 ":" _TOOL_PREFIX
           " ompt_event_master_end: codeptr_ra=%p\n",
           ompt_get_thread_data()->value, codeptr_ra);
    break;
  case ompt_scope_beginend:
    printf("ompt_scope_beginend should never be passed to %s\n", __func__);
    exit(-1);
  }
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

int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
  ompt_get_unique_id = (ompt_get_unique_id_t)lookup("ompt_get_unique_id");
  ompt_get_thread_data = (ompt_get_thread_data_t)lookup("ompt_get_thread_data");

  register_ompt_callback(ompt_callback_master);
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

// Check if libomp supports the callbacks for this test.
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_master'

// CHECK: 0: NULL_POINTER=[[NULL:.*$]]

// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_master_begin:
// CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
// CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
// CHECK: {{^}}[[MASTER_ID]]: ompt_event_master_end:
// CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_END:0x[0-f]+]]
// CHECK: {{^}}[[MASTER_ID]]: current_address={{.*}}[[RETURN_ADDRESS_END]]
