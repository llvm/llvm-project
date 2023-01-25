#include <dlfcn.h>
#include <omp-tools.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

typedef struct omp_t_data {
  // Thread data
  ompt_state_t ompt_state;
  ompt_wait_id_t ompt_wait_id;
  int omp_thread_num;
  ompt_data_t *ompt_thread_data;
  // Parallel data
  int omp_num_threads;
  int omp_level;
  int omp_active_level;
  ompt_data_t *ompt_parallel_data;
  // Task data
  int omp_max_threads;
  int omp_parallel;
  int omp_final;
  int omp_dynamic;
  int omp_nested;
  int omp_max_active_levels;
  omp_sched_t omp_kind;
  int omp_modifier;
  omp_proc_bind_t omp_proc_bind;
  ompt_frame_t *ompt_frame_list;
  ompt_data_t *ompt_task_data;
} omp_t_data_t;

static __thread omp_t_data_t thread_data;

static ompt_function_lookup_t ompt_lookup;
// NOLINTNEXTLINE "Used in Macro:register_callback_t below."
static ompt_set_callback_t ompt_set_callback;
static ompt_get_callback_t ompt_get_callback;
static ompt_get_state_t ompt_get_state;
static ompt_get_task_info_t ompt_get_task_info;
static ompt_get_thread_data_t ompt_get_thread_data;
static ompt_get_parallel_info_t ompt_get_parallel_info;
static ompt_get_unique_id_t ompt_get_unique_id;
static ompt_get_num_procs_t ompt_get_num_procs;
static ompt_get_num_places_t ompt_get_num_places;
static ompt_get_place_proc_ids_t ompt_get_place_proc_ids;
static ompt_get_place_num_t ompt_get_place_num;
static ompt_get_partition_place_nums_t ompt_get_partition_place_nums;
static ompt_get_proc_id_t ompt_get_proc_id;
static ompt_enumerate_states_t ompt_enumerate_states;
static ompt_enumerate_mutex_impls_t ompt_enumerate_mutex_impls;
static int checks = 0;

static void on_ompt_callback_implicit_task(ompt_scope_endpoint_t endpoint,
                                           ompt_data_t *parallel_data,
                                           ompt_data_t *task_data,
                                           unsigned int team_size,
                                           unsigned int thread_num, int flags) {
  if (endpoint == ompt_scope_begin)
    task_data->value = ompt_get_unique_id();
}

static void on_ompt_callback_thread_begin(ompt_thread_t thread_type,
                                          ompt_data_t *t_data) {
  t_data->value = ompt_get_unique_id();
}

static void on_ompt_callback_parallel_begin(
    ompt_data_t *encountering_task_data,
    const ompt_frame_t *encountering_task_frame, ompt_data_t *parallel_data,
    uint32_t requested_team_size, int flag, const void *codeptr_ra) {
  parallel_data->value = ompt_get_unique_id();
}

#define register_callback_t(name, type)                                        \
  do {                                                                         \
    type f_##name = &on_##name;                                                \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_callback(name) register_callback_t(name, name##_t)

static int ompt_initialize(ompt_function_lookup_t lookup,
                           int initial_device_num, ompt_data_t *tool_data) {
  ompt_lookup = lookup;
  // TODO: remove: printf("runtime_version: %s, omp_version: %i\n",
  // runtime_version, omp_version);

  // TODO: remove macro
  // #define declare_inquery_fn(F) F = (F##_t)lookup(#F);
  // FOREACH_OMPT_INQUIRY_FN(declare_inquery_fn)
  // #undef declare_inquery_fn

  ompt_set_callback_t ompt_set_callback =
      (ompt_set_callback_t)lookup("ompt_set_callback");
  ompt_get_callback = (ompt_get_callback_t)lookup("ompt_get_callback");
  ompt_get_state = (ompt_get_state_t)lookup("ompt_get_state");
  ompt_get_task_info = (ompt_get_task_info_t)lookup("ompt_get_task_info");
  ompt_get_thread_data = (ompt_get_thread_data_t)lookup("ompt_get_thread_data");
  ompt_get_parallel_info =
      (ompt_get_parallel_info_t)lookup("ompt_get_parallel_info");
  ompt_get_unique_id = (ompt_get_unique_id_t)lookup("ompt_get_unique_id");

  ompt_get_num_procs = (ompt_get_num_procs_t)lookup("ompt_get_num_procs");
  ompt_get_num_places = (ompt_get_num_places_t)lookup("ompt_get_num_places");
  ompt_get_place_proc_ids =
      (ompt_get_place_proc_ids_t)lookup("ompt_get_place_proc_ids");
  ompt_get_place_num = (ompt_get_place_num_t)lookup("ompt_get_place_num");
  ompt_get_partition_place_nums =
      (ompt_get_partition_place_nums_t)lookup("ompt_get_partition_place_nums");
  ompt_get_proc_id = (ompt_get_proc_id_t)lookup("ompt_get_proc_id");
  ompt_enumerate_states =
      (ompt_enumerate_states_t)lookup("ompt_enumerate_states");
  ompt_enumerate_mutex_impls =
      (ompt_enumerate_mutex_impls_t)lookup("ompt_enumerate_mutex_impls");

  register_callback(ompt_callback_implicit_task);
  register_callback(ompt_callback_thread_begin);
  register_callback(ompt_callback_parallel_begin);

  return 1; // activate tool
}

static void ompt_finalize(ompt_data_t *tool_data) {}

// "This func will be invoked by OpenMP implementation, refer spec: 4.2.1"
// NOLINTNEXTLINE
static ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                                 const char *runtime_version) {
  static ompt_start_tool_result_t ompt_start_tool_result = {
      &ompt_initialize, &ompt_finalize, {0}};
  return &ompt_start_tool_result;
}

static void collectParallelData(omp_t_data_t *data) {
  data->omp_num_threads = omp_get_num_threads();
  data->omp_level = omp_get_level();
  data->omp_active_level = omp_get_active_level();
  ompt_get_parallel_info(0, &(data->ompt_parallel_data), NULL);
}

static void collectTaskData(omp_t_data_t *data) {
  data->omp_max_threads = omp_get_max_threads();
  data->omp_parallel = omp_in_parallel();
  data->omp_final = omp_in_final();
  data->omp_dynamic = omp_get_dynamic();
  data->omp_nested = omp_get_max_active_levels() > 1;
  data->omp_max_active_levels = omp_get_max_active_levels();
  omp_get_schedule(&(data->omp_kind), &(data->omp_modifier));
  data->omp_proc_bind = omp_get_proc_bind();
  ompt_get_task_info(0, NULL, &(data->ompt_task_data), &(data->ompt_frame_list),
                     NULL, NULL);
}

static void collectThreadData(omp_t_data_t *data) {
  data->omp_thread_num = omp_get_thread_num();
  data->ompt_state = (ompt_state_t)ompt_get_state(&(data->ompt_wait_id));
  data->ompt_thread_data = ompt_get_thread_data();
}

#ifdef __cplusplus
extern "C" {
#endif
__attribute__((noinline)) static void *breakToolTest(omp_t_data_t *data) {
  return data;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
static void *ompd_tool_break(void *n) {
  (void)n;
  asm("");
  return NULL;
}
#ifdef __cplusplus
}
#endif

// NOLINTNEXTLINE "This func will be invoked in testcases."
static void *ompd_tool_test(void *n) {
  collectThreadData(&thread_data);
  collectParallelData(&thread_data);
  collectTaskData(&thread_data);
  breakToolTest(&thread_data);
  checks++;
  ompd_tool_break(NULL);
  return NULL;
}

__attribute__((__constructor__)) static void init(void) {}

__attribute__((__destructor__)) static void fini(void) {
  printf("Finished %i testsuites.\n", checks);
}
