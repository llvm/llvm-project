#include "omp-tools.h"

#define ompt_start_tool disable_ompt_start_tool
#define _TOOL_PREFIX " _first_tool:"
#include "callback.h"
#undef _TOOL_PREFIX
#undef ompt_start_tool

#define CLIENT_TOOL_LIBRARIES_VAR "CUSTOM_DATA_STORAGE_TOOL_LIBRARIES"
static ompt_data_t *custom_get_client_ompt_data(ompt_data_t *);
static void free_data_pair(ompt_data_t *);
#define OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA custom_get_client_ompt_data
#define OMPT_MULTIPLEX_CUSTOM_DELETE_THREAD_DATA free_data_pair
#define OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA                         \
  custom_get_client_ompt_data
#define OMPT_MULTIPLEX_CUSTOM_DELETE_PARALLEL_DATA free_data_pair
#define OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA custom_get_client_ompt_data
#define OMPT_MULTIPLEX_CUSTOM_DELETE_TASK_DATA free_data_pair
#include "ompt-multiplex.h"

typedef struct custom_data_pair_s {
  ompt_data_t own_data;
  ompt_data_t client_data;
} custom_data_pair_t;

static ompt_data_t *custom_get_client_ompt_data(ompt_data_t *data) {
  if (data)
    return &(((custom_data_pair_t *)(data->ptr))->client_data);
  else
    return NULL;
}

static ompt_data_t *get_own_ompt_data(ompt_data_t *data) {
  if (data)
    return &(((custom_data_pair_t *)(data->ptr))->own_data);
  else
    return NULL;
}

static ompt_multiplex_data_pair_t *
allocate_data_pair(ompt_data_t *data_pointer) {
  data_pointer->ptr = malloc(sizeof(ompt_multiplex_data_pair_t));
  if (!data_pointer->ptr) {
    printf("Malloc ERROR\n");
    exit(-1);
  }
  ompt_multiplex_data_pair_t *data_pair =
      (ompt_multiplex_data_pair_t *)data_pointer->ptr;
  data_pair->own_data.ptr = NULL;
  data_pair->client_data.ptr = NULL;
  return data_pair;
}

static void free_data_pair(ompt_data_t *data_pointer) {
  free((*data_pointer).ptr);
}

static void on_cds_ompt_callback_sync_region(ompt_sync_region_t kind,
                                             ompt_scope_endpoint_t endpoint,
                                             ompt_data_t *parallel_data,
                                             ompt_data_t *task_data,
                                             const void *codeptr_ra) {
  parallel_data = get_own_ompt_data(parallel_data);
  task_data = get_own_ompt_data(task_data);
  on_ompt_callback_sync_region(kind, endpoint, parallel_data, task_data,
                               codeptr_ra);
}

static void on_cds_ompt_callback_sync_region_wait(
    ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data, ompt_data_t *task_data,
    const void *codeptr_ra) {
  parallel_data = get_own_ompt_data(parallel_data);
  task_data = get_own_ompt_data(task_data);
  on_ompt_callback_sync_region_wait(kind, endpoint, parallel_data, task_data,
                                    codeptr_ra);
}

static void on_cds_ompt_callback_flush(ompt_data_t *thread_data,
                                       const void *codeptr_ra) {
  thread_data = get_own_ompt_data(thread_data);
  on_cds_ompt_callback_flush(thread_data, codeptr_ra);
}

static void on_cds_ompt_callback_cancel(ompt_data_t *task_data, int flags,
                                        const void *codeptr_ra) {
  task_data = get_own_ompt_data(task_data);
  on_ompt_callback_cancel(task_data, flags, codeptr_ra);
}

static void on_cds_ompt_callback_implicit_task(ompt_scope_endpoint_t endpoint,
                                               ompt_data_t *parallel_data,
                                               ompt_data_t *task_data,
                                               unsigned int team_size,
                                               unsigned int thread_num,
                                               int type) {
  if (endpoint == ompt_scope_begin && (type & ompt_task_initial)) {
    allocate_data_pair(parallel_data);
  }
  if (endpoint == ompt_scope_begin) {
    allocate_data_pair(task_data);
  }
  parallel_data = get_own_ompt_data(parallel_data);
  task_data = get_own_ompt_data(task_data);
  on_ompt_callback_implicit_task(endpoint, parallel_data, task_data, team_size,
                                 thread_num, type);
}

static void on_cds_ompt_callback_work(ompt_work_t wstype,
                                      ompt_scope_endpoint_t endpoint,
                                      ompt_data_t *parallel_data,
                                      ompt_data_t *task_data, uint64_t count,
                                      const void *codeptr_ra) {
  parallel_data = get_own_ompt_data(parallel_data);
  task_data = get_own_ompt_data(task_data);
  on_ompt_callback_work(wstype, endpoint, parallel_data, task_data, count,
                        codeptr_ra);
}

static void on_cds_ompt_callback_master(ompt_scope_endpoint_t endpoint,
                                        ompt_data_t *parallel_data,
                                        ompt_data_t *task_data,
                                        const void *codeptr_ra) {
  parallel_data = get_own_ompt_data(parallel_data);
  task_data = get_own_ompt_data(task_data);
  on_ompt_callback_masked(endpoint, parallel_data, task_data, codeptr_ra);
}

static void on_cds_ompt_callback_parallel_begin(
    ompt_data_t *parent_task_data, const ompt_frame_t *parent_task_frame,
    ompt_data_t *parallel_data, uint32_t requested_team_size, int invoker,
    const void *codeptr_ra) {
  parent_task_data = get_own_ompt_data(parent_task_data);
  if (parallel_data->ptr)
    printf("%s\n", "0: parallel_data initially not null");
  allocate_data_pair(parallel_data);
  parallel_data = get_own_ompt_data(parallel_data);
  on_ompt_callback_parallel_begin(parent_task_data, parent_task_frame,
                                  parallel_data, requested_team_size, invoker,
                                  codeptr_ra);
}

static void on_cds_ompt_callback_parallel_end(ompt_data_t *parallel_data,
                                              ompt_data_t *task_data,
                                              int invoker,
                                              const void *codeptr_ra) {
  task_data = get_own_ompt_data(task_data);
  parallel_data = get_own_ompt_data(parallel_data);
  on_ompt_callback_parallel_end(parallel_data, task_data, invoker, codeptr_ra);
}

static void on_cds_ompt_callback_task_create(ompt_data_t *parent_task_data,
                                             const ompt_frame_t *parent_frame,
                                             ompt_data_t *new_task_data,
                                             int type, int has_dependences,
                                             const void *codeptr_ra) {
  parent_task_data = get_own_ompt_data(parent_task_data);
  if (new_task_data->ptr)
    printf("%s\n", "0: new_task_data initially not null");
  allocate_data_pair(new_task_data);
  new_task_data = get_own_ompt_data(new_task_data);
  on_ompt_callback_task_create(parent_task_data, parent_frame, new_task_data,
                               type, has_dependences, codeptr_ra);
}

static void
on_cds_ompt_callback_task_schedule(ompt_data_t *first_task_data,
                                   ompt_task_status_t prior_task_status,
                                   ompt_data_t *second_task_data) {
  ompt_data_t *original_first_task_data = first_task_data;
  first_task_data = get_own_ompt_data(first_task_data);
  second_task_data = get_own_ompt_data(second_task_data);
  on_ompt_callback_task_schedule(first_task_data, prior_task_status,
                                 second_task_data);
}

static void on_cds_ompt_callback_dependences(ompt_data_t *task_data,
                                             const ompt_dependence_t *deps,
                                             int ndeps) {
  task_data = get_own_ompt_data(task_data);
  on_ompt_callback_dependences(task_data, deps, ndeps);
}

static void
on_cds_ompt_callback_task_dependence(ompt_data_t *first_task_data,
                                     ompt_data_t *second_task_data) {
  first_task_data = get_own_ompt_data(first_task_data);
  second_task_data = get_own_ompt_data(second_task_data);
  on_ompt_callback_task_dependence(first_task_data, second_task_data);
}

static void on_cds_ompt_callback_thread_begin(ompt_thread_t thread_type,
                                              ompt_data_t *thread_data) {
  if (thread_data->ptr)
    printf("%s\n", "0: thread_data initially not null");
  allocate_data_pair(thread_data);
  thread_data = get_own_ompt_data(thread_data);
  on_ompt_callback_thread_begin(thread_type, thread_data);
}

static void on_cds_ompt_callback_thread_end(ompt_data_t *thread_data) {
  thread_data = get_own_ompt_data(thread_data);
  on_ompt_callback_thread_end(thread_data);
}

static int on_cds_ompt_callback_control_tool(uint64_t command,
                                             uint64_t modifier, void *arg,
                                             const void *codeptr_ra) {
  printf("%" PRIu64 ": _first_tool: ompt_event_control_tool: command=%" PRIu64
         ", modifier=%" PRIu64 ", arg=%p, codeptr_ra=%p \n",
         ompt_get_thread_data()->value, command, modifier, arg, codeptr_ra);

  // print task data
  int task_level = 0;
  ompt_data_t *task_data;
  while (ompt_get_task_info(task_level, NULL, (ompt_data_t **)&task_data, NULL,
                            NULL, NULL)) {
    task_data = get_own_ompt_data(task_data);
    printf("%" PRIu64 ": _first_tool: task level %d: task_id=%" PRIx64 "\n",
           ompt_get_thread_data()->value, task_level, task_data->value);
    task_level++;
  }

  // print parallel data
  int parallel_level = 0;
  ompt_data_t *parallel_data;
  while (ompt_get_parallel_info(parallel_level, (ompt_data_t **)&parallel_data,
                                NULL)) {
    parallel_data = get_own_ompt_data(parallel_data);
    printf("%" PRIu64 ": _first_tool: parallel level %d: parallel_id=%" PRIx64
           "\n",
           ompt_get_thread_data()->value, parallel_level, parallel_data->value);
    parallel_level++;
  }
  return 0; // success
}

static ompt_get_thread_data_t ompt_cds_get_thread_data;
ompt_data_t *ompt_get_own_thread_data() {
  return get_own_ompt_data(ompt_cds_get_thread_data());
}

#define register_ompt_callback2_t(name, type)                                  \
  do {                                                                         \
    type f_##name = &on_cds_##name;                                            \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_ompt_callback2(name) register_ompt_callback2_t(name, name##_t)

int ompt_cds_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                        ompt_data_t *tool_data) {
  ompt_initialize(lookup, initial_device_num, tool_data);
  ompt_cds_get_thread_data = ompt_get_thread_data;
  ompt_get_thread_data = ompt_get_own_thread_data;

  register_ompt_callback(ompt_callback_mutex_acquire);
  register_ompt_callback_t(ompt_callback_mutex_acquired, ompt_callback_mutex_t);
  register_ompt_callback_t(ompt_callback_mutex_released, ompt_callback_mutex_t);
  register_ompt_callback(ompt_callback_nest_lock);
  register_ompt_callback2(ompt_callback_sync_region);
  register_ompt_callback2_t(ompt_callback_sync_region_wait,
                            ompt_callback_sync_region_t);
  register_ompt_callback2(ompt_callback_control_tool);
  register_ompt_callback2(ompt_callback_flush);
  register_ompt_callback2(ompt_callback_cancel);
  register_ompt_callback2(ompt_callback_implicit_task);
  register_ompt_callback_t(ompt_callback_lock_init,
                           ompt_callback_mutex_acquire_t);
  register_ompt_callback_t(ompt_callback_lock_destroy, ompt_callback_mutex_t);
  register_ompt_callback2(ompt_callback_work);
  register_ompt_callback2(ompt_callback_master);
  register_ompt_callback2(ompt_callback_parallel_begin);
  register_ompt_callback2(ompt_callback_parallel_end);
  register_ompt_callback2(ompt_callback_task_create);
  register_ompt_callback2(ompt_callback_task_schedule);
  register_ompt_callback2(ompt_callback_dependences);
  register_ompt_callback2(ompt_callback_task_dependence);
  register_ompt_callback2(ompt_callback_thread_begin);
  register_ompt_callback2(ompt_callback_thread_end);
  return 1; // success
}

void ompt_cds_finalize(ompt_data_t *tool_data) {
  printf("0: ompt_event_runtime_shutdown\n");
}

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t ompt_start_tool_result = {
      &ompt_cds_initialize, &ompt_cds_finalize, 0};
  return &ompt_start_tool_result;
}
