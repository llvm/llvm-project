#define _BSD_SOURCE
#include <stdio.h>
#include <inttypes.h>
#include <omp.h>
#include <ompt.h>
#include "ompt-signal.h"

// Used to detect architecture
#include "../../src/kmp_platform.h"

static const char* ompt_thread_type_t_values[] = {
  NULL,
  "ompt_thread_initial",
  "ompt_thread_worker",
  "ompt_thread_other"
};

static const char* ompt_task_status_t_values[] = {
  NULL,
  "ompt_task_complete",
  "ompt_task_yield",
  "ompt_task_cancel",
  "ompt_task_others"
};
static const char* ompt_cancel_flag_t_values[] = {
  "ompt_cancel_parallel",
  "ompt_cancel_sections",
  "ompt_cancel_do",
  "ompt_cancel_taskgroup",
  "ompt_cancel_activated",
  "ompt_cancel_detected",
  "ompt_cancel_discarded_task"
};

static ompt_set_callback_t ompt_set_callback;
static ompt_get_task_info_t ompt_get_task_info;
static ompt_get_thread_data_t ompt_get_thread_data;
static ompt_get_parallel_info_t ompt_get_parallel_info;
static ompt_get_unique_id_t ompt_get_unique_id;
static ompt_get_num_places_t ompt_get_num_places;
static ompt_get_place_proc_ids_t ompt_get_place_proc_ids;
static ompt_get_place_num_t ompt_get_place_num;
static ompt_get_partition_place_nums_t ompt_get_partition_place_nums;
static ompt_get_proc_id_t ompt_get_proc_id;
static ompt_enumerate_states_t ompt_enumerate_states;
static ompt_enumerate_mutex_impls_t ompt_enumerate_mutex_impls;

static void print_ids(int level)
{
  ompt_frame_t* frame ;
  ompt_data_t* parallel_data;
  ompt_data_t* task_data;
  int exists_task = ompt_get_task_info(level, NULL, &task_data, &frame, &parallel_data, NULL);
  if (frame)
  {
    printf("%" PRIu64 ": task level %d: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", exit_frame=%p, reenter_frame=%p\n", ompt_get_thread_data()->value, level, exists_task ? parallel_data->value : 0, exists_task ? task_data->value : 0, frame->exit_frame, frame->enter_frame);
  }
  else
    printf("%" PRIu64 ": task level %d: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", frame=%p\n", ompt_get_thread_data()->value, level, exists_task ? parallel_data->value : 0, exists_task ? task_data->value : 0, frame);
}

#define print_frame(level)\
do {\
  printf("%" PRIu64 ": __builtin_frame_address(%d)=%p\n", ompt_get_thread_data()->value, level, __builtin_frame_address(level));\
} while(0)

// clang (version 5.0 and above) adds an intermediate function call with debug flag (-g)
#if defined(TEST_NEED_PRINT_FRAME_FROM_OUTLINED_FN)
  #if defined(DEBUG) && defined(__clang__) && __clang_major__ >= 5
    #define print_frame_from_outlined_fn(level) print_frame(level+1)
  #else
    #define print_frame_from_outlined_fn(level) print_frame(level)
  #endif

  #warning "Clang 5.0 and later add an additional wrapper function for tasks when compiling with debug information."
  #warning "Please define -DDEBUG iff you manually pass in -g!"
#endif

// This macro helps to define a label at the current position that can be used
// to get the current address in the code.
//
// For print_current_address():
//   To reliably determine the offset between the address of the label and the
//   actual return address, we insert a NOP instruction as a jump target as the
//   compiler would otherwise insert an instruction that we can't control. The
//   instruction length is target dependent and is explained below.
//
// (The empty block between "#pragma omp ..." and the __asm__ statement is a
// workaround for a bug in the Intel Compiler.)
#define define_ompt_label(id) \
  {} \
  __asm__("nop"); \
ompt_label_##id:

// This macro helps to get the address of a label that is inserted by the above
// macro define_ompt_label(). The address is obtained with a GNU extension
// (&&label) that has been tested with gcc, clang and icc.
#define get_ompt_label_address(id) (&& ompt_label_##id)

// This macro prints the exact address that a previously called runtime function
// returns to.
#define print_current_address(id) \
  define_ompt_label(id) \
  print_possible_return_addresses(get_ompt_label_address(id))

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
// On X86 the NOP instruction is 1 byte long. In addition, the comiler inserts
// a MOV instruction for non-void runtime functions which is 3 bytes long.
#define print_possible_return_addresses(addr) \
  printf("%" PRIu64 ": current_address=%p or %p for non-void functions\n", \
         ompt_get_thread_data()->value, ((char *)addr) - 1, ((char *)addr) - 4)
#elif KMP_ARCH_PPC64
// On Power the NOP instruction is 4 bytes long. In addition, the compiler
// inserts an LD instruction which accounts for another 4 bytes. In contrast to
// X86 this instruction is always there, even for void runtime functions.
#define print_possible_return_addresses(addr) \
  printf("%" PRIu64 ": current_address=%p\n", ompt_get_thread_data()->value, \
         ((char *)addr) - 8)
#elif KMP_ARCH_AARCH64
// On AArch64 the NOP instruction is 4 bytes long, can be followed by inserted
// store instruction (another 4 bytes long).
#define print_possible_return_addresses(addr) \
  printf("%" PRIu64 ": current_address=%p or %p\n", ompt_get_thread_data()->value, \
         ((char *)addr) - 4, ((char *)addr) - 8)
#else
#error Unsupported target architecture, cannot determine address offset!
#endif


// This macro performs a somewhat similar job to print_current_address(), except
// that it discards a certain number of nibbles from the address and only prints
// the most significant bits / nibbles. This can be used for cases where the
// return address can only be approximated.
//
// To account for overflows (ie the most significant bits / nibbles have just
// changed as we are a few bytes above the relevant power of two) the addresses
// of the "current" and of the "previous block" are printed.
#define print_fuzzy_address(id) \
  define_ompt_label(id) \
  print_fuzzy_address_blocks(get_ompt_label_address(id))

// If you change this define you need to adapt all capture patterns in the tests
// to include or discard the new number of nibbles!
#define FUZZY_ADDRESS_DISCARD_NIBBLES 2
#define FUZZY_ADDRESS_DISCARD_BYTES (1 << ((FUZZY_ADDRESS_DISCARD_NIBBLES) * 4))
#define print_fuzzy_address_blocks(addr) \
  printf("%" PRIu64 ": fuzzy_address=0x%" PRIx64 " or 0x%" PRIx64 " (%p)\n", \
  ompt_get_thread_data()->value, \
  ((uint64_t)addr) / FUZZY_ADDRESS_DISCARD_BYTES - 1, \
  ((uint64_t)addr) / FUZZY_ADDRESS_DISCARD_BYTES, addr)


static void format_task_type(int type, char* buffer)
{
  char* progress = buffer;
  if(type & ompt_task_initial) progress += sprintf(progress, "ompt_task_initial");
  if(type & ompt_task_implicit) progress += sprintf(progress, "ompt_task_implicit");
  if(type & ompt_task_explicit) progress += sprintf(progress, "ompt_task_explicit");
  if(type & ompt_task_target) progress += sprintf(progress, "ompt_task_target");
  if(type & ompt_task_undeferred) progress += sprintf(progress, "|ompt_task_undeferred");
  if(type & ompt_task_untied) progress += sprintf(progress, "|ompt_task_untied");
  if(type & ompt_task_final) progress += sprintf(progress, "|ompt_task_final");
  if(type & ompt_task_mergeable) progress += sprintf(progress, "|ompt_task_mergeable");
  if(type & ompt_task_merged) progress += sprintf(progress, "|ompt_task_merged");
}

static void
on_ompt_callback_mutex_acquire(
  ompt_mutex_kind_t kind,
  unsigned int hint,
  unsigned int impl,
  ompt_wait_id_t wait_id,
  const void *codeptr_ra)
{
  switch(kind)
  {
    case ompt_mutex_lock:
      printf("%" PRIu64 ": ompt_event_wait_lock: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra);
      break;
    case ompt_mutex_nest_lock:
      printf("%" PRIu64 ": ompt_event_wait_nest_lock: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra);
      break;
    case ompt_mutex_critical:
      printf("%" PRIu64 ": ompt_event_wait_critical: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra);
      break;
    case ompt_mutex_atomic:
      printf("%" PRIu64 ": ompt_event_wait_atomic: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra);
      break;
    case ompt_mutex_ordered:
      printf("%" PRIu64 ": ompt_event_wait_ordered: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra);
      break;
    default:
      break;
  }
}

static void
on_ompt_callback_mutex_acquired(
  ompt_mutex_kind_t kind,
  ompt_wait_id_t wait_id,
  const void *codeptr_ra)
{
  switch(kind)
  {
    case ompt_mutex_lock:
      printf("%" PRIu64 ": ompt_event_acquired_lock: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_mutex_nest_lock:
      printf("%" PRIu64 ": ompt_event_acquired_nest_lock_first: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_mutex_critical:
      printf("%" PRIu64 ": ompt_event_acquired_critical: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_mutex_atomic:
      printf("%" PRIu64 ": ompt_event_acquired_atomic: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_mutex_ordered:
      printf("%" PRIu64 ": ompt_event_acquired_ordered: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    default:
      break;
  }
}

static void
on_ompt_callback_mutex_released(
  ompt_mutex_kind_t kind,
  ompt_wait_id_t wait_id,
  const void *codeptr_ra)
{
  switch(kind)
  {
    case ompt_mutex_lock:
      printf("%" PRIu64 ": ompt_event_release_lock: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_mutex_nest_lock:
      printf("%" PRIu64 ": ompt_event_release_nest_lock_last: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_mutex_critical:
      printf("%" PRIu64 ": ompt_event_release_critical: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_mutex_atomic:
      printf("%" PRIu64 ": ompt_event_release_atomic: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_mutex_ordered:
      printf("%" PRIu64 ": ompt_event_release_ordered: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    default:
      break;
  }
}

static void
on_ompt_callback_nest_lock(
    ompt_scope_endpoint_t endpoint,
    ompt_wait_id_t wait_id,
    const void *codeptr_ra)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      printf("%" PRIu64 ": ompt_event_acquired_nest_lock_next: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_scope_end:
      printf("%" PRIu64 ": ompt_event_release_nest_lock_prev: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
  }
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
      switch(kind)
      {
        case ompt_sync_region_barrier:
          printf("%" PRIu64 ": ompt_event_barrier_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          print_ids(0);
          break;
        case ompt_sync_region_taskwait:
          printf("%" PRIu64 ": ompt_event_taskwait_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskgroup:
          printf("%" PRIu64 ": ompt_event_taskgroup_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          break;
      }
      break;
    case ompt_scope_end:
      switch(kind)
      {
        case ompt_sync_region_barrier:
          printf("%" PRIu64 ": ompt_event_barrier_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskwait:
          printf("%" PRIu64 ": ompt_event_taskwait_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskgroup:
          printf("%" PRIu64 ": ompt_event_taskgroup_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
      }
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
      switch(kind)
      {
        case ompt_sync_region_barrier:
          printf("%" PRIu64 ": ompt_event_wait_barrier_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskwait:
          printf("%" PRIu64 ": ompt_event_wait_taskwait_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskgroup:
          printf("%" PRIu64 ": ompt_event_wait_taskgroup_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
          break;
      }
      break;
    case ompt_scope_end:
      switch(kind)
      {
        case ompt_sync_region_barrier:
          printf("%" PRIu64 ": ompt_event_wait_barrier_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskwait:
          printf("%" PRIu64 ": ompt_event_wait_taskwait_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
        case ompt_sync_region_taskgroup:
          printf("%" PRIu64 ": ompt_event_wait_taskgroup_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, codeptr_ra);
          break;
      }
      break;
  }
}

static void
on_ompt_callback_flush(
    ompt_data_t *thread_data,
    const void *codeptr_ra)
{
  printf("%" PRIu64 ": ompt_event_flush: codeptr_ra=%p\n", thread_data->value, codeptr_ra);
}

static void
on_ompt_callback_cancel(
    ompt_data_t *task_data,
    int flags,
    const void *codeptr_ra)
{
  const char* first_flag_value;
  const char* second_flag_value;
  if(flags & ompt_cancel_parallel)
    first_flag_value = ompt_cancel_flag_t_values[0];
  else if(flags & ompt_cancel_sections)
    first_flag_value = ompt_cancel_flag_t_values[1];
  else if(flags & ompt_cancel_do)
    first_flag_value = ompt_cancel_flag_t_values[2];
  else if(flags & ompt_cancel_taskgroup)
    first_flag_value = ompt_cancel_flag_t_values[3];

  if(flags & ompt_cancel_activated)
    second_flag_value = ompt_cancel_flag_t_values[4];
  else if(flags & ompt_cancel_detected)
    second_flag_value = ompt_cancel_flag_t_values[5];
  else if(flags & ompt_cancel_discarded_task)
    second_flag_value = ompt_cancel_flag_t_values[6];

  printf("%" PRIu64 ": ompt_event_cancel: task_data=%" PRIu64 ", flags=%s|%s=%" PRIu32 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, task_data->value, first_flag_value, second_flag_value, flags,  codeptr_ra);
}

static void
on_ompt_callback_idle(
  ompt_scope_endpoint_t endpoint)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      printf("%" PRIu64 ": ompt_event_idle_begin:\n", ompt_get_thread_data()->value);
      break;
    case ompt_scope_end:
      printf("%" PRIu64 ": ompt_event_idle_end:\n", ompt_get_thread_data()->value);
      break;
  }
}

static void
on_ompt_callback_implicit_task(
    ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data,
    ompt_data_t *task_data,
    unsigned int team_size,
    unsigned int thread_num)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      if(task_data->ptr)
        printf("%s\n", "0: task_data initially not null");
      task_data->value = ompt_get_unique_id();
      printf("%" PRIu64 ": ompt_event_implicit_task_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", team_size=%" PRIu32 ", thread_num=%" PRIu32 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, team_size, thread_num);
      break;
    case ompt_scope_end:
      printf("%" PRIu64 ": ompt_event_implicit_task_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", team_size=%" PRIu32 ", thread_num=%" PRIu32 "\n", ompt_get_thread_data()->value, (parallel_data)?parallel_data->value:0, task_data->value, team_size, thread_num);
      break;
  }
}

static void
on_ompt_callback_lock_init(
  ompt_mutex_kind_t kind,
  unsigned int hint,
  unsigned int impl,
  ompt_wait_id_t wait_id,
  const void *codeptr_ra)
{
  switch(kind)
  {
    case ompt_mutex_lock:
      printf("%" PRIu64 ": ompt_event_init_lock: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra);
      break;
    case ompt_mutex_nest_lock:
      printf("%" PRIu64 ": ompt_event_init_nest_lock: wait_id=%" PRIu64 ", hint=%" PRIu32 ", impl=%" PRIu32 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, hint, impl, codeptr_ra);
      break;
    default:
      break;
  }
}

static void
on_ompt_callback_lock_destroy(
  ompt_mutex_kind_t kind,
  ompt_wait_id_t wait_id,
  const void *codeptr_ra)
{
  switch(kind)
  {
    case ompt_mutex_lock:
      printf("%" PRIu64 ": ompt_event_destroy_lock: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    case ompt_mutex_nest_lock:
      printf("%" PRIu64 ": ompt_event_destroy_nest_lock: wait_id=%" PRIu64 ", codeptr_ra=%p \n", ompt_get_thread_data()->value, wait_id, codeptr_ra);
      break;
    default:
      break;
  }
}

static void
on_ompt_callback_work(
  ompt_work_type_t wstype,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  uint64_t count,
  const void *codeptr_ra)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      switch(wstype)
      {
        case ompt_work_loop:
          printf("%" PRIu64 ": ompt_event_loop_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_sections:
          printf("%" PRIu64 ": ompt_event_sections_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_single_executor:
          printf("%" PRIu64 ": ompt_event_single_in_block_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_single_other:
          printf("%" PRIu64 ": ompt_event_single_others_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_workshare:
          //impl
          break;
        case ompt_work_distribute:
          printf("%" PRIu64 ": ompt_event_distribute_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_taskloop:
          //impl
          printf("%" PRIu64 ": ompt_event_taskloop_begin: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
      }
      break;
    case ompt_scope_end:
      switch(wstype)
      {
        case ompt_work_loop:
          printf("%" PRIu64 ": ompt_event_loop_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_sections:
          printf("%" PRIu64 ": ompt_event_sections_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_single_executor:
          printf("%" PRIu64 ": ompt_event_single_in_block_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_single_other:
          printf("%" PRIu64 ": ompt_event_single_others_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_workshare:
          //impl
          break;
        case ompt_work_distribute:
          printf("%" PRIu64 ": ompt_event_distribute_end: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
        case ompt_work_taskloop:
          //impl
          printf("%" PRIu64 ": ompt_event_taskloop_end: parallel_id=%" PRIu64 ", parent_task_id=%" PRIu64 ", codeptr_ra=%p, count=%" PRIu64 "\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra, count);
          break;
      }
      break;
  }
}

static void
on_ompt_callback_master(
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra)
{
  switch(endpoint)
  {
    case ompt_scope_begin:
      printf("%" PRIu64 ": ompt_event_master_begin: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
      break;
    case ompt_scope_end:
      printf("%" PRIu64 ": ompt_event_master_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, task_data->value, codeptr_ra);
      break;
  }
}

static void
on_ompt_callback_parallel_begin(
  ompt_data_t *encountering_task_data,
  const ompt_frame_t *encountering_task_frame,
  ompt_data_t* parallel_data,
  uint32_t requested_team_size,
  ompt_invoker_t invoker,
  const void *codeptr_ra)
{
  if(parallel_data->ptr)
    printf("0: parallel_data initially not null\n");
  parallel_data->value = ompt_get_unique_id();
  printf("%" PRIu64 ": ompt_event_parallel_begin: parent_task_id=%" PRIu64 ", parent_task_frame.exit=%p, parent_task_frame.reenter=%p, parallel_id=%" PRIu64 ", requested_team_size=%" PRIu32 ", codeptr_ra=%p, invoker=%d\n", ompt_get_thread_data()->value, encountering_task_data->value, encountering_task_frame->exit_frame, encountering_task_frame->enter_frame, parallel_data->value, requested_team_size, codeptr_ra, invoker);
}

static void
on_ompt_callback_parallel_end(
  ompt_data_t *parallel_data,
  ompt_data_t *encountering_task_data,
  ompt_invoker_t invoker,
  const void *codeptr_ra)
{
  printf("%" PRIu64 ": ompt_event_parallel_end: parallel_id=%" PRIu64 ", task_id=%" PRIu64 ", invoker=%d, codeptr_ra=%p\n", ompt_get_thread_data()->value, parallel_data->value, encountering_task_data->value, invoker, codeptr_ra);
}

static void
on_ompt_callback_task_create(
    ompt_data_t *encountering_task_data,
    const ompt_frame_t *encountering_task_frame,
    ompt_data_t* new_task_data,
    int type,
    int has_dependences,
    const void *codeptr_ra)
{
  if(new_task_data->ptr)
    printf("0: new_task_data initially not null\n");
  new_task_data->value = ompt_get_unique_id();
  char buffer[2048];

  format_task_type(type, buffer);

  //there is no parallel_begin callback for implicit parallel region
  //thus it is initialized in initial task
  if(type & ompt_task_initial)
  {
    ompt_data_t *parallel_data;
    ompt_get_parallel_info(0, &parallel_data, NULL);
    if(parallel_data->ptr)
      printf("%s\n", "0: parallel_data initially not null");
    parallel_data->value = ompt_get_unique_id();
  }

  printf("%" PRIu64 ": ompt_event_task_create: parent_task_id=%" PRIu64 ", parent_task_frame.exit=%p, parent_task_frame.reenter=%p, new_task_id=%" PRIu64 ", codeptr_ra=%p, task_type=%s=%d, has_dependences=%s\n", ompt_get_thread_data()->value, encountering_task_data ? encountering_task_data->value : 0, encountering_task_frame ? encountering_task_frame->exit_frame : NULL, encountering_task_frame ? encountering_task_frame->enter_frame : NULL, new_task_data->value, codeptr_ra, buffer, type, has_dependences ? "yes" : "no");
}

static void
on_ompt_callback_task_schedule(
    ompt_data_t *first_task_data,
    ompt_task_status_t prior_task_status,
    ompt_data_t *second_task_data)
{
  printf("%" PRIu64 ": ompt_event_task_schedule: first_task_id=%" PRIu64 ", second_task_id=%" PRIu64 ", prior_task_status=%s=%d\n", ompt_get_thread_data()->value, first_task_data->value, second_task_data->value, ompt_task_status_t_values[prior_task_status], prior_task_status);
  if(prior_task_status == ompt_task_complete)
  {
    printf("%" PRIu64 ": ompt_event_task_end: task_id=%" PRIu64 "\n", ompt_get_thread_data()->value, first_task_data->value);
  }
}

static void
on_ompt_callback_task_dependences(
  ompt_data_t *task_data,
  const ompt_task_dependence_t *deps,
  int ndeps)
{
  printf("%" PRIu64 ": ompt_event_task_dependences: task_id=%" PRIu64 ", deps=%p, ndeps=%d\n", ompt_get_thread_data()->value, task_data->value, (void *)deps, ndeps);
}

static void
on_ompt_callback_task_dependence(
  ompt_data_t *first_task_data,
  ompt_data_t *second_task_data)
{
  printf("%" PRIu64 ": ompt_event_task_dependence_pair: first_task_id=%" PRIu64 ", second_task_id=%" PRIu64 "\n", ompt_get_thread_data()->value, first_task_data->value, second_task_data->value);
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
on_ompt_callback_thread_end(
  ompt_data_t *thread_data)
{
  printf("%" PRIu64 ": ompt_event_thread_end: thread_id=%" PRIu64 "\n", ompt_get_thread_data()->value, thread_data->value);
}

static int
on_ompt_callback_control_tool(
  uint64_t command,
  uint64_t modifier,
  void *arg,
  const void *codeptr_ra)
{
  ompt_frame_t* omptTaskFrame;
  ompt_get_task_info(0, NULL, (ompt_data_t**) NULL, &omptTaskFrame, NULL, NULL);
  printf("%" PRIu64 ": ompt_event_control_tool: command=%" PRIu64 ", modifier=%" PRIu64 ", arg=%p, codeptr_ra=%p, current_task_frame.exit=%p, current_task_frame.reenter=%p \n", ompt_get_thread_data()->value, command, modifier, arg, codeptr_ra, omptTaskFrame->exit_frame, omptTaskFrame->enter_frame);
  return 0; //success
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
  ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  ompt_get_task_info = (ompt_get_task_info_t) lookup("ompt_get_task_info");
  ompt_get_thread_data = (ompt_get_thread_data_t) lookup("ompt_get_thread_data");
  ompt_get_parallel_info = (ompt_get_parallel_info_t) lookup("ompt_get_parallel_info");
  ompt_get_unique_id = (ompt_get_unique_id_t) lookup("ompt_get_unique_id");

  ompt_get_num_places = (ompt_get_num_places_t) lookup("ompt_get_num_places");
  ompt_get_place_proc_ids = (ompt_get_place_proc_ids_t) lookup("ompt_get_place_proc_ids");
  ompt_get_place_num = (ompt_get_place_num_t) lookup("ompt_get_place_num");
  ompt_get_partition_place_nums = (ompt_get_partition_place_nums_t) lookup("ompt_get_partition_place_nums");
  ompt_get_proc_id = (ompt_get_proc_id_t) lookup("ompt_get_proc_id");
  ompt_enumerate_states = (ompt_enumerate_states_t) lookup("ompt_enumerate_states");
  ompt_enumerate_mutex_impls = (ompt_enumerate_mutex_impls_t) lookup("ompt_enumerate_mutex_impls");

  register_callback(ompt_callback_mutex_acquire);
  register_callback_t(ompt_callback_mutex_acquired, ompt_callback_mutex_t);
  register_callback_t(ompt_callback_mutex_released, ompt_callback_mutex_t);
  register_callback(ompt_callback_nest_lock);
  register_callback(ompt_callback_sync_region);
  register_callback_t(ompt_callback_sync_region_wait, ompt_callback_sync_region_t);
  register_callback(ompt_callback_control_tool);
  register_callback(ompt_callback_flush);
  register_callback(ompt_callback_cancel);
  register_callback(ompt_callback_idle);
  register_callback(ompt_callback_implicit_task);
  register_callback_t(ompt_callback_lock_init, ompt_callback_mutex_acquire_t);
  register_callback_t(ompt_callback_lock_destroy, ompt_callback_mutex_t);
  register_callback(ompt_callback_work);
  register_callback(ompt_callback_master);
  register_callback(ompt_callback_parallel_begin);
  register_callback(ompt_callback_parallel_end);
  register_callback(ompt_callback_task_create);
  register_callback(ompt_callback_task_schedule);
  register_callback(ompt_callback_task_dependences);
  register_callback(ompt_callback_task_dependence);
  register_callback(ompt_callback_thread_begin);
  register_callback(ompt_callback_thread_end);
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
