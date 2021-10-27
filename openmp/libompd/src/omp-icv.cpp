/*
 * omp-icv.cpp -- OMPD Internal Control Variable handling
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// clang-format off
/* clang-format expect kmp.h before omp.h which results in build break
 * due to a few redeclarations.
 */
#include "omp-debug.h"
// NOLINTNEXTLINE "to avoid clang tidy warning for the same reason as above."
#include "omp.h"
#include "ompd-private.h"
#include "TargetValue.h"
#include "kmp.h"
#include <cstring>

/* The ICVs ompd-final-var and ompd-implicit-var below are for backward
 * compatibility with 5.0.
 */

#define FOREACH_OMPD_ICV(macro)                                                     \
  macro(dyn_var, "dyn-var", ompd_scope_thread, 0)                                   \
  macro(run_sched_var, "run-sched-var", ompd_scope_task, 0)                         \
  macro(stacksize_var, "stacksize-var", ompd_scope_address_space, 0)                \
  macro(cancel_var, "cancel-var", ompd_scope_address_space, 0)                      \
  macro(max_task_priority_var, "max-task-priority-var", ompd_scope_address_space, 0)\
  macro(debug_var, "debug-var", ompd_scope_address_space, 0)                        \
  macro(nthreads_var, "nthreads-var", ompd_scope_thread, 0)                         \
  macro(display_affinity_var, "display-affinity-var", ompd_scope_address_space, 0)  \
  macro(affinity_format_var, "affinity-format-var", ompd_scope_address_space, 0)    \
  macro(default_device_var, "default-device-var", ompd_scope_thread, 0)             \
  macro(tool_var, "tool-var", ompd_scope_address_space, 0)                          \
  macro(tool_libraries_var, "tool-libraries-var", ompd_scope_address_space, 0)      \
  macro(tool_verbose_init_var, "tool-verbose-init-var", ompd_scope_address_space, 0)\
  macro(levels_var, "levels-var", ompd_scope_parallel, 1)                           \
  macro(active_levels_var, "active-levels-var", ompd_scope_parallel, 0)             \
  macro(thread_limit_var, "thread-limit-var", ompd_scope_task, 0)                   \
  macro(max_active_levels_var, "max-active-levels-var", ompd_scope_task, 0)         \
  macro(bind_var, "bind-var", ompd_scope_task, 0)                                   \
  macro(num_procs_var, "num-procs-var", ompd_scope_address_space, 0)                \
  macro(ompd_num_procs_var, "ompd-num-procs-var", ompd_scope_address_space, 0)      \
  macro(thread_num_var, "thread-num-var", ompd_scope_thread, 1)                     \
  macro(ompd_thread_num_var, "ompd-thread-num-var", ompd_scope_thread, 1)           \
  macro(final_var, "final-task-var", ompd_scope_task, 0)                            \
  macro(ompd_final_var, "ompd-final-var", ompd_scope_task, 0)                       \
  macro(ompd_final_task_var, "ompd-final-task-var", ompd_scope_task, 0)             \
  macro(implicit_var, "implicit-task-var", ompd_scope_task, 0)                      \
  macro(ompd_implicit_var, "ompd-implicit-var", ompd_scope_task, 0)                 \
  macro(ompd_implicit_task_var, "ompd-implicit-task-var", ompd_scope_task, 0)       \
  macro(team_size_var, "team-size-var", ompd_scope_parallel, 1)                     \
  macro(ompd_team_size_var, "ompd-team-size-var", ompd_scope_parallel, 1)

void __ompd_init_icvs(const ompd_callbacks_t *table) { callbacks = table; }

enum ompd_icv {
  ompd_icv_undefined_marker =
      0, // ompd_icv_undefined is already defined in ompd.h
#define ompd_icv_macro(v, n, s, d) ompd_icv_##v,
  FOREACH_OMPD_ICV(ompd_icv_macro)
#undef ompd_icv_macro
      ompd_icv_after_last_icv
};

static const char *ompd_icv_string_values[] = {"undefined",
#define ompd_icv_macro(v, n, s, d) n,
  FOREACH_OMPD_ICV(ompd_icv_macro)
#undef ompd_icv_macro
};

static const ompd_scope_t ompd_icv_scope_values[] = {
    ompd_scope_global, // undefined marker
#define ompd_icv_macro(v, n, s, d) s,
    FOREACH_OMPD_ICV(ompd_icv_macro)
#undef ompd_icv_macro
};

// clang-format on
ompd_rc_t ompd_enumerate_icvs(ompd_address_space_handle_t *handle,
                              ompd_icv_id_t current, ompd_icv_id_t *next_id,
                              const char **next_icv_name,
                              ompd_scope_t *next_scope, int *more) {
  if (!handle) {
    return ompd_rc_stale_handle;
  }
  if (!next_id || !next_icv_name || !next_scope || !more) {
    return ompd_rc_bad_input;
  }
  if (current + 1 >= ompd_icv_after_last_icv) {
    return ompd_rc_bad_input;
  }

  *next_id = current + 1;

  char *icv_name = NULL;
  ompd_rc_t ret = callbacks->alloc_memory(
      std::strlen(ompd_icv_string_values[*next_id]) + 1, (void **)&icv_name);
  *next_icv_name = icv_name;
  if (ret != ompd_rc_ok) {
    return ret;
  }
  std::strcpy(icv_name, ompd_icv_string_values[*next_id]);

  *next_scope = ompd_icv_scope_values[*next_id];

  if ((*next_id) + 1 >= ompd_icv_after_last_icv) {
    *more = 0;
  } else {
    *more = 1;
  }

  return ompd_rc_ok;
}

static ompd_rc_t create_empty_string(const char **empty_string_ptr) {
  char *empty_str;
  ompd_rc_t ret;

  if (!callbacks) {
    return ompd_rc_callback_error;
  }
  ret = callbacks->alloc_memory(1, (void **)&empty_str);
  if (ret != ompd_rc_ok) {
    return ret;
  }
  empty_str[0] = '\0';
  *empty_string_ptr = empty_str;
  return ompd_rc_ok;
}

static ompd_rc_t ompd_get_dynamic(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle */
    ompd_word_t *dyn_val /* OUT: Dynamic adjustment of threads */
) {
  if (!thread_handle)
    return ompd_rc_stale_handle;
  if (!thread_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  int8_t dynamic;
  ompd_rc_t ret =
      TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
          .cast("kmp_base_info_t")
          .access("th_current_task") /*__kmp_threads[t]->th.th_current_task*/
          .cast("kmp_taskdata_t", 1)
          .access("td_icvs") /*__kmp_threads[t]->th.th_current_task->td_icvs*/
          .cast("kmp_internal_control_t", 0)
          .access(
              "dynamic") /*__kmp_threads[t]->th.th_current_task->td_icvs.dynamic*/
          .castBase()
          .getValue(dynamic);
  *dyn_val = dynamic;
  return ret;
}

static ompd_rc_t
ompd_get_stacksize(ompd_address_space_handle_t
                       *addr_handle, /* IN: handle for the address space */
                   ompd_word_t *stacksize_val /* OUT: per thread stack size */
) {
  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;
  ompd_rc_t ret;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  size_t stacksize;
  ret = TValue(context, "__kmp_stksize")
            .castBase("__kmp_stksize")
            .getValue(stacksize);
  *stacksize_val = stacksize;
  return ret;
}

static ompd_rc_t ompd_get_cancellation(
    ompd_address_space_handle_t
        *addr_handle,             /* IN: handle for the address space */
    ompd_word_t *cancellation_val /* OUT: cancellation value */
) {
  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }
  ompd_rc_t ret;

  int omp_cancellation;
  ret = TValue(context, "__kmp_omp_cancellation")
            .castBase("__kmp_omp_cancellation")
            .getValue(omp_cancellation);
  *cancellation_val = omp_cancellation;
  return ret;
}

static ompd_rc_t ompd_get_max_task_priority(
    ompd_address_space_handle_t
        *addr_handle,                  /* IN: handle for the address space */
    ompd_word_t *max_task_priority_val /* OUT: max task priority value */
) {
  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }
  ompd_rc_t ret;

  int max_task_priority;
  ret = TValue(context, "__kmp_max_task_priority")
            .castBase("__kmp_max_task_priority")
            .getValue(max_task_priority);
  *max_task_priority_val = max_task_priority;
  return ret;
}

static ompd_rc_t
ompd_get_debug(ompd_address_space_handle_t
                   *addr_handle,      /* IN: handle for the address space */
               ompd_word_t *debug_val /* OUT: debug value */
) {
  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }
  ompd_rc_t ret;

  uint64_t ompd_state_val;
  ret = TValue(context, "ompd_state")
            .castBase("ompd_state")
            .getValue(ompd_state_val);
  if (ompd_state_val > 0) {
    *debug_val = 1;
  } else {
    *debug_val = 0;
  }
  return ret;
}

/* Helper routine for the ompd_get_nthreads routines */
static ompd_rc_t ompd_get_nthreads_aux(ompd_thread_handle_t *thread_handle,
                                       uint32_t *used,
                                       uint32_t *current_nesting_level,
                                       uint32_t *nproc) {
  if (!thread_handle)
    return ompd_rc_stale_handle;
  if (!thread_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  ompd_rc_t ret = TValue(context, "__kmp_nested_nth")
                      .cast("kmp_nested_nthreads_t")
                      .access("used")
                      .castBase(ompd_type_int)
                      .getValue(*used);
  if (ret != ompd_rc_ok)
    return ret;

  TValue taskdata =
      TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
          .cast("kmp_base_info_t")
          .access("th_current_task") /*__kmp_threads[t]->th.th_current_task*/
          .cast("kmp_taskdata_t", 1);

  ret = taskdata
            .access("td_team") /*__kmp_threads[t]->th.th_current_task.td_team*/
            .cast("kmp_team_p", 1)
            .access("t") /*__kmp_threads[t]->th.th_current_task.td_team->t*/
            .cast("kmp_base_team_t", 0) /*t*/
            .access("t_level")          /*t.t_level*/
            .castBase(ompd_type_int)
            .getValue(*current_nesting_level);
  if (ret != ompd_rc_ok)
    return ret;

  ret = taskdata.cast("kmp_taskdata_t", 1)
            .access("td_icvs") /*__kmp_threads[t]->th.th_current_task->td_icvs*/
            .cast("kmp_internal_control_t", 0)
            .access(
                "nproc") /*__kmp_threads[t]->th.th_current_task->td_icvs.nproc*/
            .castBase(ompd_type_int)
            .getValue(*nproc);
  if (ret != ompd_rc_ok)
    return ret;

  return ompd_rc_ok;
}

static ompd_rc_t ompd_get_nthreads(
    ompd_thread_handle_t *thread_handle, /* IN: handle for the thread */
    ompd_word_t *nthreads_var_val        /* OUT: nthreads-var (of integer type)
                                            value */
) {
  uint32_t used;
  uint32_t nproc;
  uint32_t current_nesting_level;

  ompd_rc_t ret;
  ret = ompd_get_nthreads_aux(thread_handle, &used, &current_nesting_level,
                              &nproc);
  if (ret != ompd_rc_ok)
    return ret;

  /*__kmp_threads[t]->th.th_current_task->td_icvs.nproc*/
  *nthreads_var_val = nproc;
  /* If the nthreads-var is a list with more than one element, then the value of
     this ICV cannot be represented by an integer type. In this case,
     ompd_rc_incomplete is returned. The tool can check the return value and
     can choose to invoke ompd_get_icv_string_from_scope() if needed. */
  if (current_nesting_level < used - 1) {
    return ompd_rc_incomplete;
  }
  return ompd_rc_ok;
}

static ompd_rc_t ompd_get_nthreads(
    ompd_thread_handle_t *thread_handle, /* IN: handle for the thread */
    const char **nthreads_list_string    /* OUT: string list of comma separated
                                            nthreads values */
) {
  uint32_t used;
  uint32_t nproc;
  uint32_t current_nesting_level;

  ompd_rc_t ret;
  ret = ompd_get_nthreads_aux(thread_handle, &used, &current_nesting_level,
                              &nproc);
  if (ret != ompd_rc_ok)
    return ret;

  uint32_t num_list_elems;
  if (used == 0 || current_nesting_level >= used) {
    num_list_elems = 1;
  } else {
    num_list_elems = used - current_nesting_level;
  }
  size_t buffer_size = 16 /* digits per element including the comma separator */
                           * num_list_elems +
                       1; /* string terminator NULL */
  char *nthreads_list_str;
  ret = callbacks->alloc_memory(buffer_size, (void **)&nthreads_list_str);
  if (ret != ompd_rc_ok)
    return ret;

  /* The nthreads-var list would be:
  [__kmp_threads[t]->th.th_current_task->td_icvs.nproc,
   __kmp_nested_nth.nth[current_nesting_level + 1],
   __kmp_nested_nth.nth[current_nesting_level + 2],
    …,
   __kmp_nested_nth.nth[used - 1]]*/

  sprintf(nthreads_list_str, "%d", nproc);
  *nthreads_list_string = nthreads_list_str;
  if (num_list_elems == 1) {
    return ompd_rc_ok;
  }

  char temp_value[16];
  uint32_t nth_value;

  for (current_nesting_level++; /* the list element for this nesting
                                 * level has already been accounted for
                                   by nproc */
       current_nesting_level < used; current_nesting_level++) {

    ret = TValue(thread_handle->ah->context, "__kmp_nested_nth")
              .cast("kmp_nested_nthreads_t")
              .access("nth")
              .cast("int", 1)
              .getArrayElement(current_nesting_level)
              .castBase(ompd_type_int)
              .getValue(nth_value);

    if (ret != ompd_rc_ok)
      return ret;

    sprintf(temp_value, ",%d", nth_value);
    strcat(nthreads_list_str, temp_value);
  }

  return ompd_rc_ok;
}

static ompd_rc_t ompd_get_display_affinity(
    ompd_address_space_handle_t
        *addr_handle,                 /* IN: handle for the address space */
    ompd_word_t *display_affinity_val /* OUT: display affinity value */
) {
  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;
  ompd_rc_t ret;

  if (!callbacks) {
    return ompd_rc_callback_error;
  }
  ret = TValue(context, "__kmp_display_affinity")
            .castBase("__kmp_display_affinity")
            .getValue(*display_affinity_val);
  return ret;
}

static ompd_rc_t ompd_get_affinity_format(
    ompd_address_space_handle_t *addr_handle, /* IN: address space handle*/
    const char **affinity_format_string       /* OUT: affinity format string */
) {
  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;

  if (!callbacks) {
    return ompd_rc_callback_error;
  }
  ompd_rc_t ret;
  ret = TValue(context, "__kmp_affinity_format")
            .cast("char", 1)
            .getString(affinity_format_string);
  return ret;
}

static ompd_rc_t ompd_get_tool_libraries(
    ompd_address_space_handle_t *addr_handle, /* IN: address space handle*/
    const char **tool_libraries_string        /* OUT: tool libraries string */
) {
  if (!tool_libraries_string)
    return ompd_rc_bad_input;

  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;

  if (!callbacks) {
    return ompd_rc_callback_error;
  }
  ompd_rc_t ret;
  ret = TValue(context, "__kmp_tool_libraries")
            .cast("char", 1)
            .getString(tool_libraries_string);
  if (ret == ompd_rc_unsupported) {
    ret = create_empty_string(tool_libraries_string);
  }
  return ret;
}

static ompd_rc_t ompd_get_default_device(
    ompd_thread_handle_t *thread_handle, /* IN: handle for the thread */
    ompd_word_t *default_device_val      /* OUT: default device value */
) {
  if (!thread_handle)
    return ompd_rc_stale_handle;
  if (!thread_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks)
    return ompd_rc_callback_error;

  ompd_rc_t ret =
      TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
          .cast("kmp_base_info_t")
          .access("th_current_task") /*__kmp_threads[t]->th.th_current_task*/
          .cast("kmp_taskdata_t", 1)
          .access("td_icvs") /*__kmp_threads[t]->th.th_current_task->td_icvs*/
          .cast("kmp_internal_control_t", 0)
          /*__kmp_threads[t]->th.th_current_task->td_icvs.default_device*/
          .access("default_device")
          .castBase()
          .getValue(*default_device_val);
  return ret;
}

static ompd_rc_t
ompd_get_tool(ompd_address_space_handle_t
                  *addr_handle,     /* IN: handle for the address space */
              ompd_word_t *tool_val /* OUT: tool value */
) {
  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }
  ompd_rc_t ret;

  ret =
      TValue(context, "__kmp_tool").castBase("__kmp_tool").getValue(*tool_val);
  return ret;
}

static ompd_rc_t ompd_get_tool_verbose_init(
    ompd_address_space_handle_t *addr_handle, /* IN: address space handle*/
    const char **tool_verbose_init_string /* OUT: tool verbose init string */
) {
  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;

  if (!callbacks) {
    return ompd_rc_callback_error;
  }
  ompd_rc_t ret;
  ret = TValue(context, "__kmp_tool_verbose_init")
            .cast("char", 1)
            .getString(tool_verbose_init_string);
  if (ret == ompd_rc_unsupported) {
    ret = create_empty_string(tool_verbose_init_string);
  }
  return ret;
}

static ompd_rc_t ompd_get_level(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: nesting level */
) {
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  uint32_t res;

  ompd_rc_t ret = TValue(context, parallel_handle->th)
                      .cast("kmp_base_team_t", 0) /*t*/
                      .access("t_level")          /*t.t_level*/
                      .castBase()
                      .getValue(res);
  *val = res;
  return ret;
}

static ompd_rc_t ompd_get_active_level(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: active nesting level */
) {
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  uint32_t res;

  ompd_rc_t ret = TValue(context, parallel_handle->th)
                      .cast("kmp_base_team_t", 0) /*t*/
                      .access("t_active_level")   /*t.t_active_level*/
                      .castBase()
                      .getValue(res);
  *val = res;
  return ret;
}

static ompd_rc_t
ompd_get_num_procs(ompd_address_space_handle_t
                       *addr_handle, /* IN: handle for the address space */
                   ompd_word_t *val  /* OUT: number of processes */
) {
  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  if (!val)
    return ompd_rc_bad_input;
  ompd_rc_t ret;

  int nth;
  ret = TValue(context, "__kmp_avail_proc")
            .castBase("__kmp_avail_proc")
            .getValue(nth);
  *val = nth;
  return ret;
}

static ompd_rc_t ompd_get_thread_limit(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
    ompd_word_t *val                 /* OUT: max number of threads */
) {
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  ompd_rc_t ret = TValue(context, task_handle->th)
                      .cast("kmp_taskdata_t") // td
                      .access("td_icvs")      // td->td_icvs
                      .cast("kmp_internal_control_t", 0)
                      .access("thread_limit") // td->td_icvs.thread_limit
                      .castBase()
                      .getValue(*val);

  return ret;
}

static ompd_rc_t ompd_get_thread_num(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_word_t *val /* OUT: number of the thread within the team */
) {
  if (!thread_handle)
    return ompd_rc_stale_handle;
  if (!thread_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  ompd_rc_t ret =
      TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
          .cast("kmp_base_info_t")
          .access("th_info") /*__kmp_threads[t]->th.th_info*/
          .cast("kmp_desc_t")
          .access("ds") /*__kmp_threads[t]->th.th_info.ds*/
          .cast("kmp_desc_base_t")
          .access("ds_tid") /*__kmp_threads[t]->th.th_info.ds.ds_tid*/
          .castBase()
          .getValue(*val);
  return ret;
}

static ompd_rc_t
ompd_in_final(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
              ompd_word_t *val                 /* OUT: max number of threads */
) {
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  ompd_rc_t ret = TValue(context, task_handle->th)
                      .cast("kmp_taskdata_t") // td
                      .access("td_flags")     // td->td_flags
                      .cast("kmp_tasking_flags_t")
                      .check("final", val); // td->td_flags.tasktype

  return ret;
}

static ompd_rc_t ompd_get_max_active_levels(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
    ompd_word_t *val                 /* OUT: max number of threads */
) {
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  ompd_rc_t ret =
      TValue(context, task_handle->th)
          .cast("kmp_taskdata_t") // td
          .access("td_icvs")      // td->td_icvs
          .cast("kmp_internal_control_t", 0)
          .access("max_active_levels") // td->td_icvs.max_active_levels
          .castBase()
          .getValue(*val);

  return ret;
}

static ompd_rc_t ompd_get_run_schedule(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
    const char **run_sched_string    /* OUT: Run Schedule String
                                        consisting of kind and modifier */
) {
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  int kind;

  TValue sched = TValue(context, task_handle->th)
                     .cast("kmp_taskdata_t") // td
                     .access("td_icvs")      // td->td_icvs
                     .cast("kmp_internal_control_t", 0)
                     .access("sched") // td->td_icvs.sched
                     .cast("kmp_r_sched_t", 0);

  ompd_rc_t ret = sched
                      .access("r_sched_type") // td->td_icvs.sched.r_sched_type
                      .castBase()
                      .getValue(kind);
  if (ret != ompd_rc_ok) {
    return ret;
  }
  int chunk = 0;
  ret = sched
            .access("chunk") // td->td_icvs.sched.chunk
            .castBase()
            .getValue(chunk);
  if (ret != ompd_rc_ok) {
    return ret;
  }
  char *run_sched_var_string;
  ret = callbacks->alloc_memory(100, (void **)&run_sched_var_string);
  if (ret != ompd_rc_ok) {
    return ret;
  }
  run_sched_var_string[0] = '\0';
  if (SCHEDULE_HAS_MONOTONIC(kind)) {
    strcpy(run_sched_var_string, "monotonic:");
  } else if (SCHEDULE_HAS_NONMONOTONIC(kind)) {
    strcpy(run_sched_var_string, "nonmonotonic:");
  }

  bool static_unchunked = false;
  switch (SCHEDULE_WITHOUT_MODIFIERS(kind)) {
  case kmp_sch_static:
  case kmp_sch_static_greedy:
  case kmp_sch_static_balanced:
    static_unchunked = true;
    strcat(run_sched_var_string, "static");
    break;
  case kmp_sch_static_chunked:
    strcat(run_sched_var_string, "static");
    break;
  case kmp_sch_dynamic_chunked:
    strcat(run_sched_var_string, "dynamic");
    break;
  case kmp_sch_guided_chunked:
  case kmp_sch_guided_iterative_chunked:
  case kmp_sch_guided_analytical_chunked:
    strcat(run_sched_var_string, "guided");
    break;
  case kmp_sch_auto:
    strcat(run_sched_var_string, "auto");
    break;
  case kmp_sch_trapezoidal:
    strcat(run_sched_var_string, "trapezoidal");
    break;
  case kmp_sch_static_steal:
    strcat(run_sched_var_string, "static_steal");
    break;
  default:
    ret = callbacks->free_memory((void *)(run_sched_var_string));
    if (ret != ompd_rc_ok) {
      return ret;
    }
    ret = create_empty_string(run_sched_string);
    return ret;
  }

  if (static_unchunked == true) {
    // To be in sync with what OMPT returns.
    // Chunk was not set. Shown with a zero value.
    chunk = 0;
  }

  char temp_str[16];
  sprintf(temp_str, ",%d", chunk);
  strcat(run_sched_var_string, temp_str);
  *run_sched_string = run_sched_var_string;
  return ret;
}

/* Helper routine for the ompd_get_proc_bind routines */
static ompd_rc_t ompd_get_proc_bind_aux(ompd_task_handle_t *task_handle,
                                        uint32_t *used,
                                        uint32_t *current_nesting_level,
                                        uint32_t *proc_bind) {
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  ompd_rc_t ret = TValue(context, "__kmp_nested_proc_bind")
                      .cast("kmp_nested_proc_bind_t")
                      .access("used")
                      .castBase(ompd_type_int)
                      .getValue(*used);
  if (ret != ompd_rc_ok)
    return ret;

  TValue taskdata = TValue(context, task_handle->th) /* td */
                        .cast("kmp_taskdata_t");

  ret = taskdata
            .access("td_team") /* td->td_team*/
            .cast("kmp_team_p", 1)
            .access("t")                /* td->td_team->t*/
            .cast("kmp_base_team_t", 0) /*t*/
            .access("t_level")          /*t.t_level*/
            .castBase(ompd_type_int)
            .getValue(*current_nesting_level);
  if (ret != ompd_rc_ok)
    return ret;

  ret = taskdata
            .access("td_icvs") /* td->td_icvs */
            .cast("kmp_internal_control_t", 0)
            .access("proc_bind") /* td->td_icvs.proc_bind */
            .castBase()
            .getValue(*proc_bind);
  return ret;
}

static ompd_rc_t
ompd_get_proc_bind(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle */
                   ompd_word_t *bind /* OUT: Kind of proc-binding */
) {
  uint32_t used;
  uint32_t proc_bind;
  uint32_t current_nesting_level;

  ompd_rc_t ret;
  ret = ompd_get_proc_bind_aux(task_handle, &used, &current_nesting_level,
                               &proc_bind);
  if (ret != ompd_rc_ok)
    return ret;

  *bind = proc_bind;
  /* If bind-var is a list with more than one element, then the value of
     this ICV cannot be represented by an integer type. In this case,
     ompd_rc_incomplete is returned. The tool can check the return value and
     can choose to invoke ompd_get_icv_string_from_scope() if needed. */
  if (current_nesting_level < used - 1) {
    return ompd_rc_incomplete;
  }
  return ompd_rc_ok;
}

static ompd_rc_t ompd_get_proc_bind(
    ompd_task_handle_t *task_handle,   /* IN: OpenMP task handle */
    const char **proc_bind_list_string /* OUT: string list of comma separated
                                          bind-var values */
) {
  uint32_t used;
  uint32_t proc_bind;
  uint32_t current_nesting_level;

  ompd_rc_t ret;
  ret = ompd_get_proc_bind_aux(task_handle, &used, &current_nesting_level,
                               &proc_bind);
  if (ret != ompd_rc_ok)
    return ret;

  uint32_t num_list_elems;
  if (used == 0 || current_nesting_level >= used) {
    num_list_elems = 1;
  } else {
    num_list_elems = used - current_nesting_level;
  }
  size_t buffer_size = 16 /* digits per element including the comma separator */
                           * num_list_elems +
                       1; /* string terminator NULL */
  char *proc_bind_list_str;
  ret = callbacks->alloc_memory(buffer_size, (void **)&proc_bind_list_str);
  if (ret != ompd_rc_ok)
    return ret;

  /* The bind-var list would be:
  [td->td_icvs.proc_bind,
   __kmp_nested_proc_bind.bind_types[current_nesting_level + 1],
   __kmp_nested_proc_bind.bind_types[current_nesting_level + 2],
    …,
   __kmp_nested_proc_bind.bind_types[used - 1]]*/

  sprintf(proc_bind_list_str, "%d", proc_bind);
  *proc_bind_list_string = proc_bind_list_str;
  if (num_list_elems == 1) {
    return ompd_rc_ok;
  }

  char temp_value[16];
  uint32_t bind_types_value;

  for (current_nesting_level++; /* the list element for this nesting
                                   level has already been accounted for
                                   by proc_bind */
       current_nesting_level < used; current_nesting_level++) {

    ret = TValue(task_handle->ah->context, "__kmp_nested_proc_bind")
              .cast("kmp_nested_proc_bind_t")
              .access("bind_types")
              .cast("int", 1)
              .getArrayElement(current_nesting_level)
              .castBase(ompd_type_int)
              .getValue(bind_types_value);

    if (ret != ompd_rc_ok)
      return ret;

    sprintf(temp_value, ",%d", bind_types_value);
    strcat(proc_bind_list_str, temp_value);
  }

  return ompd_rc_ok;
}

static ompd_rc_t
ompd_is_implicit(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                 ompd_word_t *val /* OUT: max number of threads */
) {
  if (!task_handle)
    return ompd_rc_stale_handle;
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  ompd_rc_t ret = TValue(context, task_handle->th)
                      .cast("kmp_taskdata_t") // td
                      .access("td_flags")     // td->td_flags
                      .cast("kmp_tasking_flags_t")
                      .check("tasktype", val); // td->td_flags.tasktype
  *val ^= 1; // tasktype: explicit = 1, implicit = 0 => invert the value
  return ret;
}

ompd_rc_t ompd_get_num_threads(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: number of threads */
) {
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  ompd_rc_t ret = ompd_rc_ok;
  if (parallel_handle->lwt.address != 0) {
    *val = 1;
  } else {
    uint32_t res;
    ret = TValue(context, parallel_handle->th)
              .cast("kmp_base_team_t", 0) /*t*/
              .access("t_nproc")          /*t.t_nproc*/
              .castBase()
              .getValue(res);
    *val = res;
  }
  return ret;
}

ompd_rc_t ompd_get_icv_from_scope(void *handle, ompd_scope_t scope,
                                  ompd_icv_id_t icv_id,
                                  ompd_word_t *icv_value) {
  if (!handle) {
    return ompd_rc_stale_handle;
  }
  if (icv_id >= ompd_icv_after_last_icv || icv_id == 0) {
    return ompd_rc_bad_input;
  }
  if (scope != ompd_icv_scope_values[icv_id]) {
    return ompd_rc_bad_input;
  }

  ompd_device_t device_kind;

  switch (scope) {
  case ompd_scope_thread:
    device_kind = ((ompd_thread_handle_t *)handle)->ah->kind;
    break;
  case ompd_scope_parallel:
    device_kind = ((ompd_parallel_handle_t *)handle)->ah->kind;
    break;
  case ompd_scope_address_space:
    device_kind = ((ompd_address_space_handle_t *)handle)->kind;
    break;
  case ompd_scope_task:
    device_kind = ((ompd_task_handle_t *)handle)->ah->kind;
    break;
  default:
    return ompd_rc_bad_input;
  }

  if (device_kind == OMPD_DEVICE_KIND_HOST) {
    switch (icv_id) {
    case ompd_icv_dyn_var:
      return ompd_get_dynamic((ompd_thread_handle_t *)handle, icv_value);
    case ompd_icv_run_sched_var:
      return ompd_rc_incompatible;
    case ompd_icv_stacksize_var:
      return ompd_get_stacksize((ompd_address_space_handle_t *)handle,
                                icv_value);
    case ompd_icv_cancel_var:
      return ompd_get_cancellation((ompd_address_space_handle_t *)handle,
                                   icv_value);
    case ompd_icv_max_task_priority_var:
      return ompd_get_max_task_priority((ompd_address_space_handle_t *)handle,
                                        icv_value);
    case ompd_icv_debug_var:
      return ompd_get_debug((ompd_address_space_handle_t *)handle, icv_value);
    case ompd_icv_nthreads_var:
      return ompd_get_nthreads((ompd_thread_handle_t *)handle, icv_value);
    case ompd_icv_display_affinity_var:
      return ompd_get_display_affinity((ompd_address_space_handle_t *)handle,
                                       icv_value);
    case ompd_icv_affinity_format_var:
      return ompd_rc_incompatible;
    case ompd_icv_tool_libraries_var:
      return ompd_rc_incompatible;
    case ompd_icv_default_device_var:
      return ompd_get_default_device((ompd_thread_handle_t *)handle, icv_value);
    case ompd_icv_tool_var:
      return ompd_get_tool((ompd_address_space_handle_t *)handle, icv_value);
    case ompd_icv_tool_verbose_init_var:
      return ompd_rc_incompatible;
    case ompd_icv_levels_var:
      return ompd_get_level((ompd_parallel_handle_t *)handle, icv_value);
    case ompd_icv_active_levels_var:
      return ompd_get_active_level((ompd_parallel_handle_t *)handle, icv_value);
    case ompd_icv_thread_limit_var:
      return ompd_get_thread_limit((ompd_task_handle_t *)handle, icv_value);
    case ompd_icv_max_active_levels_var:
      return ompd_get_max_active_levels((ompd_task_handle_t *)handle,
                                        icv_value);
    case ompd_icv_bind_var:
      return ompd_get_proc_bind((ompd_task_handle_t *)handle, icv_value);
    case ompd_icv_num_procs_var:
    case ompd_icv_ompd_num_procs_var:
      return ompd_get_num_procs((ompd_address_space_handle_t *)handle,
                                icv_value);
    case ompd_icv_thread_num_var:
    case ompd_icv_ompd_thread_num_var:
      return ompd_get_thread_num((ompd_thread_handle_t *)handle, icv_value);
    case ompd_icv_final_var:
    case ompd_icv_ompd_final_var:
    case ompd_icv_ompd_final_task_var:
      return ompd_in_final((ompd_task_handle_t *)handle, icv_value);
    case ompd_icv_implicit_var:
    case ompd_icv_ompd_implicit_var:
    case ompd_icv_ompd_implicit_task_var:
      return ompd_is_implicit((ompd_task_handle_t *)handle, icv_value);
    case ompd_icv_team_size_var:
    case ompd_icv_ompd_team_size_var:
      return ompd_get_num_threads((ompd_parallel_handle_t *)handle, icv_value);
    default:
      return ompd_rc_unsupported;
    }
  }
  return ompd_rc_unsupported;
}

ompd_rc_t ompd_get_icv_string_from_scope(void *handle, ompd_scope_t scope,
                                         ompd_icv_id_t icv_id,
                                         const char **icv_string) {
  if (!handle) {
    return ompd_rc_stale_handle;
  }
  if (icv_id >= ompd_icv_after_last_icv || icv_id == 0) {
    return ompd_rc_bad_input;
  }
  if (scope != ompd_icv_scope_values[icv_id]) {
    return ompd_rc_bad_input;
  }

  ompd_device_t device_kind;

  switch (scope) {
  case ompd_scope_thread:
    device_kind = ((ompd_thread_handle_t *)handle)->ah->kind;
    break;
  case ompd_scope_parallel:
    device_kind = ((ompd_parallel_handle_t *)handle)->ah->kind;
    break;
  case ompd_scope_address_space:
    device_kind = ((ompd_address_space_handle_t *)handle)->kind;
    break;
  case ompd_scope_task:
    device_kind = ((ompd_task_handle_t *)handle)->ah->kind;
    break;
  default:
    return ompd_rc_bad_input;
  }

  if (device_kind == OMPD_DEVICE_KIND_HOST) {
    switch (icv_id) {
    case ompd_icv_run_sched_var:
      return ompd_get_run_schedule((ompd_task_handle_t *)handle, icv_string);
    case ompd_icv_nthreads_var:
      return ompd_get_nthreads((ompd_thread_handle_t *)handle, icv_string);
    case ompd_icv_bind_var:
      return ompd_get_proc_bind((ompd_task_handle_t *)handle, icv_string);
    case ompd_icv_affinity_format_var:
      return ompd_get_affinity_format((ompd_address_space_handle_t *)handle,
                                      icv_string);
    case ompd_icv_tool_libraries_var:
      return ompd_get_tool_libraries((ompd_address_space_handle_t *)handle,
                                     icv_string);
    case ompd_icv_tool_verbose_init_var:
      return ompd_get_tool_verbose_init((ompd_address_space_handle_t *)handle,
                                        icv_string);
    default:
      return ompd_rc_unsupported;
    }
  }
  return ompd_rc_unsupported;
}

static ompd_rc_t __ompd_get_tool_data(TValue &dataValue, ompd_word_t *value,
                                      ompd_address_t *ptr) {
  ompd_rc_t ret = dataValue.getError();
  if (ret != ompd_rc_ok)
    return ret;
  ret = dataValue.access("value").castBase().getValue(*value);
  if (ret != ompd_rc_ok)
    return ret;
  ptr->segment = OMPD_SEGMENT_UNSPECIFIED;
  ret = dataValue.access("ptr").castBase().getValue(ptr->address);
  return ret;
}

ompd_rc_t ompd_get_task_data(ompd_task_handle_t *task_handle,
                             ompd_word_t *value, ompd_address_t *ptr) {
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  TValue dataValue;
  if (task_handle->lwt.address) {
    dataValue = TValue(context, task_handle->lwt)
                    .cast("ompt_lw_taskteam_t") /*lwt*/
                    .access("ompt_task_info")   // lwt->ompt_task_info
                    .cast("ompt_task_info_t")
                    .access("task_data") // lwt->ompd_task_info.task_data
                    .cast("ompt_data_t");
  } else {
    dataValue = TValue(context, task_handle->th)
                    .cast("kmp_taskdata_t")   /*td*/
                    .access("ompt_task_info") // td->ompt_task_info
                    .cast("ompt_task_info_t")
                    .access("task_data") // td->ompd_task_info.task_data
                    .cast("ompt_data_t");
  }
  return __ompd_get_tool_data(dataValue, value, ptr);
}

ompd_rc_t ompd_get_parallel_data(ompd_parallel_handle_t *parallel_handle,
                                 ompd_word_t *value, ompd_address_t *ptr) {
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  TValue dataValue;
  if (parallel_handle->lwt.address) {
    dataValue =
        TValue(context, parallel_handle->lwt)
            .cast("ompt_lw_taskteam_t") /*lwt*/
            .access("ompt_team_info")   // lwt->ompt_team_info
            .cast("ompt_team_info_t")
            .access("parallel_data") // lwt->ompt_team_info.parallel_data
            .cast("ompt_data_t");
  } else {
    dataValue = TValue(context, parallel_handle->th)
                    .cast("kmp_base_team_t")  /*t*/
                    .access("ompt_team_info") // t->ompt_team_info
                    .cast("ompt_team_info_t")
                    .access("parallel_data") // t->ompt_team_info.parallel_data
                    .cast("ompt_data_t");
  }
  return __ompd_get_tool_data(dataValue, value, ptr);
}

ompd_rc_t ompd_get_thread_data(ompd_thread_handle_t *thread_handle,
                               ompd_word_t *value, ompd_address_t *ptr) {
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!callbacks) {
    return ompd_rc_callback_error;
  }

  TValue dataValue =
      TValue(context, thread_handle->th)
          .cast("kmp_base_info_t")    /*th*/
          .access("ompt_thread_info") // th->ompt_thread_info
          .cast("ompt_thread_info_t")
          .access("thread_data") // th->ompt_thread_info.thread_data
          .cast("ompt_data_t");
  return __ompd_get_tool_data(dataValue, value, ptr);
}

ompd_rc_t ompd_get_tool_data(void *handle, ompd_scope_t scope,
                             ompd_word_t *value, ompd_address_t *ptr) {
  if (!handle) {
    return ompd_rc_stale_handle;
  }

  ompd_device_t device_kind;

  switch (scope) {
  case ompd_scope_thread:
    device_kind = ((ompd_thread_handle_t *)handle)->ah->kind;
    break;
  case ompd_scope_parallel:
    device_kind = ((ompd_parallel_handle_t *)handle)->ah->kind;
    break;
  case ompd_scope_task:
    device_kind = ((ompd_task_handle_t *)handle)->ah->kind;
    break;
  default:
    return ompd_rc_bad_input;
  }

  if (device_kind == OMPD_DEVICE_KIND_HOST) {
    switch (scope) {
    case ompd_scope_thread:
      return ompd_get_thread_data((ompd_thread_handle_t *)handle, value, ptr);
    case ompd_scope_parallel:
      return ompd_get_parallel_data((ompd_parallel_handle_t *)handle, value,
                                    ptr);
    case ompd_scope_task:
      return ompd_get_task_data((ompd_task_handle_t *)handle, value, ptr);
    default:
      return ompd_rc_unsupported;
    }
  }
  return ompd_rc_unsupported;
}
