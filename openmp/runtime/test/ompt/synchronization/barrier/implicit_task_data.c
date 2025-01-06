// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt

// This test checks that values stored in task_data in a barrier_begin event
// are still present in the corresponding barrier_end event.
// Therefore, callback implementations different from the ones in callback.h are necessary.
// This is a test for an issue reported in 
// https://github.com/OpenMPToolsInterface/LLVM-openmp/issues/39

#define _BSD_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <omp.h>
#include <omp-tools.h>

static const char* ompt_thread_t_values[] = {
  NULL,
  "ompt_thread_initial",
  "ompt_thread_worker",
  "ompt_thread_other"
};

static ompt_get_unique_id_t ompt_get_unique_id;
static ompt_get_thread_data_t ompt_get_thread_data;

int main()
{
  #pragma omp parallel num_threads(4)
  {
    #pragma omp master
    {
      sleep(1);
    }
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region_wait'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // master thread implicit barrier at parallel end
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=0, task_id=[[TASK_ID:[0-9]+]], codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_parallel_begin: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_parallel_end: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra={{0x[0-f]*}}


  // worker thread implicit barrier at parallel end
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_barrier_implicit_parallel_begin: parallel_id=0, task_id=[[TASK_ID:[0-9]+]], codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_parallel_begin: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_parallel_end: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra=[[NULL]]
  // clang-format on

  return 0;
}

static void
on_ompt_callback_thread_begin(
  ompt_thread_t thread_type,
  ompt_data_t *thread_data)
{
  if(thread_data->ptr)
    printf("%s\n", "0: thread_data initially not null");
  thread_data->value = ompt_get_unique_id();
  printf("%" PRIu64 ": ompt_event_thread_begin: thread_type=%s=%d, thread_id=%" PRIu64 "\n", ompt_get_thread_data()->value, ompt_thread_t_values[thread_type], thread_type, thread_data->value);
}

static void
on_ompt_callback_sync_region(
  ompt_sync_region_t kind,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra)
{
  // We only expect implicit parallel barrier in this code.
  if (kind != ompt_sync_region_barrier_implicit_parallel) {
    printf("unexpected ompt_sync_region_t passed to %s\n", __func__);
    exit(-1);
  }
  const char *event_name = NULL;
  if (endpoint == ompt_scope_begin) {
    event_name = "ompt_event_barrier_implicit_parallel_begin";
    task_data->value = ompt_get_unique_id();
  } else if (endpoint == ompt_scope_end) {
    event_name = "ompt_event_barrier_implicit_parallel_end";
  } else {
    printf("ompt_scope_beginend should never be passed to %s\n", __func__);
    exit(-1);
  }
  printf("%" PRIu64 ": %s: parallel_id=%" PRIu64 ", task_id=%" PRIu64
         ", codeptr_ra=%p\n",
         ompt_get_thread_data()->value, event_name,
         parallel_data ? parallel_data->value : 0, task_data->value,
         codeptr_ra);
}

static void
on_ompt_callback_sync_region_wait(
  ompt_sync_region_t kind,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra)
{
  if (kind != ompt_sync_region_barrier_implicit_parallel) {
    printf("unexpected ompt_sync_region_t passed to %s\n", __func__);
    exit(-1);
  }
  const char *event_name = NULL;
  if (endpoint == ompt_scope_begin) {
    event_name = "ompt_event_wait_barrier_implicit_parallel_begin";
  } else if (endpoint == ompt_scope_end) {
    event_name = "ompt_event_wait_barrier_implicit_parallel_end";
  } else {
    printf("ompt_scope_beginend should never be passed to %s\n", __func__);
    exit(-1);
  }
  printf("%" PRIu64 ": %s: parallel_id=%" PRIu64 ", task_id=%" PRIu64
         ", codeptr_ra=%p\n",
         ompt_get_thread_data()->value, event_name,
         parallel_data ? parallel_data->value : 0, task_data->value,
         codeptr_ra);
}

#define register_ompt_callback_t(name, type)                       \
do{                                                           \
  type f_##name = &on_##name;                                 \
  if (ompt_set_callback(name, (ompt_callback_t)f_##name) ==   \
      ompt_set_never)                                         \
    printf("0: Could not register callback '" #name "'\n");   \
}while(0)

#define register_ompt_callback(name) register_ompt_callback_t(name, name##_t)

int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_set_callback_t ompt_set_callback;
  ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  ompt_get_unique_id = (ompt_get_unique_id_t) lookup("ompt_get_unique_id");
  ompt_get_thread_data = (ompt_get_thread_data_t) lookup("ompt_get_thread_data");
  register_ompt_callback(ompt_callback_sync_region);
  register_ompt_callback_t(ompt_callback_sync_region_wait, ompt_callback_sync_region_t);
  register_ompt_callback(ompt_callback_thread_begin);
  printf("0: NULL_POINTER=%p\n", (void*)NULL);
  return 1; //success
}

void ompt_finalize(ompt_data_t *tool_data)
{
  printf("0: ompt_event_runtime_shutdown\n");
}

ompt_start_tool_result_t* ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,&ompt_finalize, 0};
  return &ompt_start_tool_result;
}
