#include <stdio.h>
#include <inttypes.h>
#include <ompt.h>

static ompt_get_task_id_t ompt_get_task_id;
static ompt_get_thread_id_t ompt_get_thread_id;
static ompt_get_parallel_id_t ompt_get_parallel_id;

static void print_ids(int level)
{
  printf("%" PRIu64 ": level %d: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_id(), level, ompt_get_parallel_id(level), ompt_get_task_id(level));
}

static void
on_ompt_event_barrier_begin(
  ompt_parallel_id_t parallel_id,
  ompt_task_id_t task_id)
{
  printf("%" PRIu64 ": ompt_event_barrier_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_id(), parallel_id, task_id);
}

static void
on_ompt_event_barrier_end(
  ompt_parallel_id_t parallel_id,
  ompt_task_id_t task_id)
{
  printf("%" PRIu64 ": ompt_event_barrier_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_id(), parallel_id, task_id);
}

static void
on_ompt_event_implicit_task_begin(
  ompt_parallel_id_t parallel_id,
  ompt_task_id_t task_id)
{
  printf("%" PRIu64 ": ompt_event_implicit_task_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_id(), parallel_id, task_id);
}

static void
on_ompt_event_implicit_task_end(
  ompt_parallel_id_t parallel_id,
  ompt_task_id_t task_id)
{
  printf("%" PRIu64 ": ompt_event_implicit_task_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_id(), parallel_id, task_id);
}

static void
on_ompt_event_loop_begin(
  ompt_parallel_id_t parallel_id,
  ompt_task_id_t parent_task_id,
  void *workshare_function)
{
  printf("%" PRIu64 ": ompt_event_loop_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", workshare_function=%p\n", ompt_get_thread_id(), parallel_id, parent_task_id, workshare_function);
}

static void
on_ompt_event_loop_end(
  ompt_parallel_id_t parallel_id,
  ompt_task_id_t task_id)
{
  printf("%" PRIu64 ": ompt_event_loop_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 "\n", ompt_get_thread_id(), parallel_id, task_id);
}

static void
on_ompt_event_parallel_begin(
  ompt_task_id_t parent_task_id,
  ompt_frame_t *parent_task_frame,
  ompt_parallel_id_t parallel_id,
  uint32_t requested_team_size,
  void *parallel_function,
  ompt_invoker_t invoker)
{
  printf("%" PRIu64 ": ompt_event_parallel_begin: parent_task_id=%" PRIu64 ", parent_task_frame=%p, parallel_id=%" PRIu64 ", requested_team_size=%" PRIu32 ", parallel_function=%p, invoker=%d\n", ompt_get_thread_id(), parent_task_id, parent_task_frame, parallel_id, requested_team_size, parallel_function, invoker);
}

static void
on_ompt_event_parallel_end(
  ompt_parallel_id_t parallel_id,
  ompt_task_id_t task_id,
  ompt_invoker_t invoker)
{
  printf("%" PRIu64 ": ompt_event_parallel_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", invoker=%d\n", ompt_get_thread_id(), parallel_id, task_id, invoker);
}


void ompt_initialize(
  ompt_function_lookup_t lookup,
  const char *runtime_version,
  unsigned int ompt_version)
{
  ompt_set_callback_t ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  ompt_get_task_id = (ompt_get_task_id_t) lookup("ompt_get_task_id");
  ompt_get_thread_id = (ompt_get_thread_id_t) lookup("ompt_get_thread_id");
  ompt_get_parallel_id = (ompt_get_parallel_id_t) lookup("ompt_get_parallel_id");

  ompt_set_callback(ompt_event_barrier_begin, (ompt_callback_t) &on_ompt_event_barrier_begin);
  ompt_set_callback(ompt_event_barrier_end, (ompt_callback_t) &on_ompt_event_barrier_end);
  ompt_set_callback(ompt_event_implicit_task_begin, (ompt_callback_t) &on_ompt_event_implicit_task_begin);
  ompt_set_callback(ompt_event_implicit_task_end, (ompt_callback_t) &on_ompt_event_implicit_task_end);
  ompt_set_callback(ompt_event_loop_begin, (ompt_callback_t) &on_ompt_event_loop_begin);
  ompt_set_callback(ompt_event_loop_end, (ompt_callback_t) &on_ompt_event_loop_end);
  ompt_set_callback(ompt_event_parallel_begin, (ompt_callback_t) &on_ompt_event_parallel_begin);
  ompt_set_callback(ompt_event_parallel_end, (ompt_callback_t) &on_ompt_event_parallel_end);
}

ompt_initialize_t ompt_tool()
{
  return &ompt_initialize;
}
