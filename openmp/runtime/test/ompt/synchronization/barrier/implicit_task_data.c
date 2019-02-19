// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt

// This test checks that values stored in task_data in a barrier_begin event
// are still present in the corresponding barrier_end event.
// Therefore, callback implementations different from the ones in callback.h are neccessary.
// This is a test for an issue reported in 
// https://github.com/OpenMPToolsInterface/LLVM-openmp/issues/39

#define _BSD_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <omp.h>
#include <ompt.h>

static const char* ompt_thread_type_t_values[] = {
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


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region_wait'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // master thread implicit barrier at parallel end
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_barrier_begin: parallel_id=0, task_id=[[TASK_ID:[0-9]+]], codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_begin: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_end: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_end: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra={{0x[0-f]*}}


  // worker thread implicit barrier at parallel end
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_barrier_begin: parallel_id=0, task_id=[[TASK_ID:[0-9]+]], codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_begin: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_end: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id=0, task_id=[[TASK_ID]], codeptr_ra=[[NULL]]

  return 0;
}

static void
on_ompt_callback_thread_begin(
  ompt_thread_type_t thread_type,
  ompt_data_t *thread_data)
{
  if(thread_data->ptr)
    printf("%s\n", "0: thread_data initially not null");
  thread_data->value = ompt_get_unique_id();
  printf("%" PRIu64 ": ompt_event_thread_begin: thread_type=%s=%d, thread_id=%" PRIu64 "\n", ompt_get_thread_data()->value, ompt_thread_type_t_values[thread_type], thread_type, thread_data->value);
}

static void
on_ompt_callback_sync_region(
  ompt_sync_region_kind_t kind,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      task_data->value = ompt_get_unique_id();
      if(kind == ompt_sync_region_barrier)
        printf("%" PRIu64 ": ompt_event_barrier_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
      break;
    case ompt_scope_end:
      if(kind == ompt_sync_region_barrier)
        printf("%" PRIu64 ": ompt_event_barrier_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
      break;
  }
}

static void
on_ompt_callback_sync_region_wait(
  ompt_sync_region_kind_t kind,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      if(kind == ompt_sync_region_barrier)
          printf("%" PRIu64 ": ompt_event_wait_barrier_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
      break;
    case ompt_scope_end:
      if(kind == ompt_sync_region_barrier)
        printf("%" PRIu64 ": ompt_event_wait_barrier_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
      break;
  }
}

#define register_callback_t(name, type)                       \
do{                                                           \
  type f_##name = &on_##name;                                 \
  if (ompt_set_callback(name, (ompt_callback_t)f_##name) ==   \
      ompt_set_never)                                         \
    printf("0: Could not register callback '" #name "'\n");   \
}while(0)

#define register_callback(name) register_callback_t(name, name##_t)

int ompt_initialize(
  ompt_function_lookup_t lookup,
  ompt_data_t *tool_data)
{
  ompt_set_callback_t ompt_set_callback;
  ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  ompt_get_unique_id = (ompt_get_unique_id_t) lookup("ompt_get_unique_id");
  ompt_get_thread_data = (ompt_get_thread_data_t) lookup("ompt_get_thread_data");
  register_callback(ompt_callback_sync_region);
  register_callback_t(ompt_callback_sync_region_wait, ompt_callback_sync_region_t);
  register_callback(ompt_callback_thread_begin);
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
