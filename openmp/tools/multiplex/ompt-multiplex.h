//===--- ompt-multiplex.h - header-only multiplexing of OMPT tools -- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file enables an OMPT tool to load another OMPT tool and
// automatically forwards OMPT event-callbacks to the nested tool.
//
// For details see openmp/tools/multiplex/README.md
//
//===----------------------------------------------------------------------===//

#ifndef OMPT_MULTIPLEX_H
#define OMPT_MULTIPLEX_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <execinfo.h>
#include <inttypes.h>
#include <omp-tools.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

static ompt_set_callback_t ompt_multiplex_set_callback;
static ompt_get_task_info_t ompt_multiplex_get_task_info;
static ompt_get_thread_data_t ompt_multiplex_get_thread_data;
static ompt_get_parallel_info_t ompt_multiplex_get_parallel_info;

// contains name of the environment var in which the tool path is specified
#ifndef CLIENT_TOOL_LIBRARIES_VAR
#error CLIENT_TOOL_LIBRARIES_VAR should be defined before including of ompt-multiplex.h
#endif

#if defined(CUSTOM_DELETE_DATA) && !defined(CUSTOM_GET_CLIENT_DATA)
#error CUSTOM_GET_CLIENT_DATA must be set if CUSTOM_DELETE_DATA is set
#endif

#define OMPT_API_ROUTINE static

#define OMPT_LOAD_CLIENT_FOREACH_OMPT_EVENT(macro)                             \
  macro(callback_thread_begin, ompt_callback_thread_begin_t, 1);               \
  macro(callback_thread_end, ompt_callback_thread_end_t, 2);                   \
  macro(callback_parallel_begin, ompt_callback_parallel_begin_t, 3);           \
  macro(callback_parallel_end, ompt_callback_parallel_end_t, 4);               \
  macro(callback_task_create, ompt_callback_task_create_t, 5);                 \
  macro(callback_task_schedule, ompt_callback_task_schedule_t, 6);             \
  macro(callback_implicit_task, ompt_callback_implicit_task_t, 7);             \
  macro(callback_target, ompt_callback_target_t, 8);                           \
  macro(callback_target_data_op, ompt_callback_target_data_op_t, 9);           \
  macro(callback_target_submit, ompt_callback_target_submit_t, 10);            \
  macro(callback_control_tool, ompt_callback_control_tool_t, 11);              \
  macro(callback_device_initialize, ompt_callback_device_initialize_t, 12);    \
  macro(callback_device_finalize, ompt_callback_device_finalize_t, 13);        \
  macro(callback_device_load, ompt_callback_device_load_t, 14);                \
  macro(callback_device_unload, ompt_callback_device_unload_t, 15);            \
  macro(callback_sync_region_wait, ompt_callback_sync_region_t, 16);           \
  macro(callback_mutex_released, ompt_callback_mutex_t, 17);                   \
  macro(callback_dependences, ompt_callback_dependences_t, 18);                \
  macro(callback_task_dependence, ompt_callback_task_dependence_t, 19);        \
  macro(callback_work, ompt_callback_work_t, 20);                              \
  macro(callback_master, ompt_callback_master_t, 21);                          \
  macro(callback_target_map, ompt_callback_target_map_t, 22);                  \
  macro(callback_sync_region, ompt_callback_sync_region_t, 23);                \
  macro(callback_lock_init, ompt_callback_mutex_acquire_t, 24);                \
  macro(callback_lock_destroy, ompt_callback_mutex_t, 25);                     \
  macro(callback_mutex_acquire, ompt_callback_mutex_acquire_t, 26);            \
  macro(callback_mutex_acquired, ompt_callback_mutex_t, 27);                   \
  macro(callback_nest_lock, ompt_callback_nest_lock_t, 28);                    \
  macro(callback_flush, ompt_callback_flush_t, 29);                            \
  macro(callback_cancel, ompt_callback_cancel_t, 30);                          \
  macro(callback_reduction, ompt_callback_sync_region_t, 31);                  \
  macro(callback_dispatch, ompt_callback_dispatch_t, 32);

typedef struct ompt_multiplex_callbacks_s {
#define ompt_event_macro(event, callback, eventid) callback ompt_##event

  OMPT_LOAD_CLIENT_FOREACH_OMPT_EVENT(ompt_event_macro)

#undef ompt_event_macro
} ompt_multiplex_callbacks_t;

typedef struct ompt_multiplex_callback_implementation_status_s {
#define ompt_event_macro(event, callback, eventid) int ompt_##event

  OMPT_LOAD_CLIENT_FOREACH_OMPT_EVENT(ompt_event_macro)

#undef ompt_event_macro
} ompt_multiplex_callback_implementation_status_t;

ompt_start_tool_result_t *ompt_multiplex_own_fns;
ompt_start_tool_result_t *ompt_multiplex_client_fns;
ompt_function_lookup_t ompt_multiplex_lookup_function;
ompt_multiplex_callbacks_t ompt_multiplex_own_callbacks,
    ompt_multiplex_client_callbacks;
ompt_multiplex_callback_implementation_status_t
    ompt_multiplex_implementation_status;

typedef struct ompt_multiplex_data_pair_s {
  ompt_data_t own_data;
  ompt_data_t client_data;
} ompt_multiplex_data_pair_t;

#if !defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA) ||                  \
    !defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA) ||                \
    !defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA)
static ompt_multiplex_data_pair_t *
ompt_multiplex_allocate_data_pair(ompt_data_t *data_pointer) {
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

static void ompt_multiplex_free_data_pair(ompt_data_t *data_pointer) {
  free((*data_pointer).ptr);
}

static ompt_data_t *ompt_multiplex_get_own_ompt_data(ompt_data_t *data) {
  if (!data)
    return NULL;
  ompt_multiplex_data_pair_t *data_pair =
      (ompt_multiplex_data_pair_t *)data->ptr;
  return &(data_pair->own_data);
}

static ompt_data_t *ompt_multiplex_get_client_ompt_data(ompt_data_t *data) {
  if (!data)
    return NULL;
  ompt_multiplex_data_pair_t *data_pair =
      (ompt_multiplex_data_pair_t *)data->ptr;
  return &(data_pair->client_data);
}
#endif //! defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA) ||
       //! !defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA) ||
       //! !defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA)

static ompt_data_t *ompt_multiplex_get_own_thread_data(ompt_data_t *data) {
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA
  return ompt_multiplex_get_own_ompt_data(data);
#else
  return data;
#endif
}

static ompt_data_t *ompt_multiplex_get_own_parallel_data(ompt_data_t *data) {
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA
  return ompt_multiplex_get_own_ompt_data(data);
#else
  return data;
#endif
}

static ompt_data_t *ompt_multiplex_get_own_task_data(ompt_data_t *data) {
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA
  return ompt_multiplex_get_own_ompt_data(data);
#else
  return data;
#endif
}

static ompt_data_t *ompt_multiplex_get_client_thread_data(ompt_data_t *data) {
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA
  return ompt_multiplex_get_client_ompt_data(data);
#else
  return OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA(data);
#endif
}

static ompt_data_t *ompt_multiplex_get_client_parallel_data(ompt_data_t *data) {
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA
  return ompt_multiplex_get_client_ompt_data(data);
#else
  return OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA(data);
#endif
}

static ompt_data_t *ompt_multiplex_get_client_task_data(ompt_data_t *data) {
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA
  return ompt_multiplex_get_client_ompt_data(data);
#else
  return OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA(data);
#endif
}

static void ompt_multiplex_callback_mutex_acquire(ompt_mutex_t kind,
                                                  unsigned int hint,
                                                  unsigned int impl,
                                                  ompt_wait_id_t wait_id,
                                                  const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_mutex_acquire) {
    ompt_multiplex_own_callbacks.ompt_callback_mutex_acquire(
        kind, hint, impl, wait_id, codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_mutex_acquire) {
    ompt_multiplex_client_callbacks.ompt_callback_mutex_acquire(
        kind, hint, impl, wait_id, codeptr_ra);
  }
}

static void ompt_multiplex_callback_mutex_acquired(ompt_mutex_t kind,
                                                   ompt_wait_id_t wait_id,
                                                   const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_mutex_acquired) {
    ompt_multiplex_own_callbacks.ompt_callback_mutex_acquired(kind, wait_id,
                                                              codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_mutex_acquired) {
    ompt_multiplex_client_callbacks.ompt_callback_mutex_acquired(kind, wait_id,
                                                                 codeptr_ra);
  }
}

static void ompt_multiplex_callback_mutex_released(ompt_mutex_t kind,
                                                   ompt_wait_id_t wait_id,
                                                   const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_mutex_released) {
    ompt_multiplex_own_callbacks.ompt_callback_mutex_released(kind, wait_id,
                                                              codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_mutex_released) {
    ompt_multiplex_client_callbacks.ompt_callback_mutex_released(kind, wait_id,
                                                                 codeptr_ra);
  }
}

static void ompt_multiplex_callback_nest_lock(ompt_scope_endpoint_t endpoint,
                                              ompt_wait_id_t wait_id,
                                              const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_nest_lock) {
    ompt_multiplex_own_callbacks.ompt_callback_nest_lock(endpoint, wait_id,
                                                         codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_nest_lock) {
    ompt_multiplex_client_callbacks.ompt_callback_nest_lock(endpoint, wait_id,
                                                            codeptr_ra);
  }
}

static void ompt_multiplex_callback_sync_region(ompt_sync_region_t kind,
                                                ompt_scope_endpoint_t endpoint,
                                                ompt_data_t *parallel_data,
                                                ompt_data_t *task_data,
                                                const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_sync_region) {
    ompt_multiplex_own_callbacks.ompt_callback_sync_region(
        kind, endpoint, ompt_multiplex_get_own_parallel_data(parallel_data),
        ompt_multiplex_get_own_task_data(task_data), codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_sync_region) {
    ompt_multiplex_client_callbacks.ompt_callback_sync_region(
        kind, endpoint, ompt_multiplex_get_client_parallel_data(parallel_data),
        ompt_multiplex_get_client_task_data(task_data), codeptr_ra);
  }
}

static void ompt_multiplex_callback_sync_region_wait(
    ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data, ompt_data_t *task_data,
    const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_sync_region_wait) {
    ompt_multiplex_own_callbacks.ompt_callback_sync_region_wait(
        kind, endpoint, ompt_multiplex_get_own_parallel_data(parallel_data),
        ompt_multiplex_get_own_task_data(task_data), codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_sync_region_wait) {
    ompt_multiplex_client_callbacks.ompt_callback_sync_region_wait(
        kind, endpoint, ompt_multiplex_get_client_parallel_data(parallel_data),
        ompt_multiplex_get_client_task_data(task_data), codeptr_ra);
  }
}

static void ompt_multiplex_callback_flush(ompt_data_t *thread_data,
                                          const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_flush) {
    ompt_multiplex_own_callbacks.ompt_callback_flush(
        ompt_multiplex_get_own_thread_data(thread_data), codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_flush) {
    ompt_multiplex_client_callbacks.ompt_callback_flush(
        ompt_multiplex_get_client_thread_data(thread_data), codeptr_ra);
  }
}

static void ompt_multiplex_callback_cancel(ompt_data_t *task_data, int flags,
                                           const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_cancel) {
    ompt_multiplex_own_callbacks.ompt_callback_cancel(
        ompt_multiplex_get_own_task_data(task_data), flags, codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_cancel) {
    ompt_multiplex_client_callbacks.ompt_callback_cancel(
        ompt_multiplex_get_client_task_data(task_data), flags, codeptr_ra);
  }
}

static void ompt_multiplex_callback_implicit_task(
    ompt_scope_endpoint_t endpoint, ompt_data_t *parallel_data,
    ompt_data_t *task_data, unsigned int team_size, unsigned int thread_num,
    int flags) {
  if (endpoint == ompt_scope_begin) {
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA
    ompt_multiplex_allocate_data_pair(task_data);
#endif
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA
    if (flags & ompt_task_initial)
      ompt_multiplex_allocate_data_pair(parallel_data);
#endif
    if (ompt_multiplex_own_callbacks.ompt_callback_implicit_task) {
      ompt_multiplex_own_callbacks.ompt_callback_implicit_task(
          endpoint, ompt_multiplex_get_own_parallel_data(parallel_data),
          ompt_multiplex_get_own_task_data(task_data), team_size, thread_num,
          flags);
    }
    if (ompt_multiplex_client_callbacks.ompt_callback_implicit_task) {
      ompt_multiplex_client_callbacks.ompt_callback_implicit_task(
          endpoint, ompt_multiplex_get_client_parallel_data(parallel_data),
          ompt_multiplex_get_client_task_data(task_data), team_size, thread_num,
          flags);
    }
  } else {
// defines to make sure, callbacks are called in correct order depending on
// defines set by the user
#if defined(OMPT_MULTIPLEX_CUSTOM_DELETE_TASK_DATA) ||                         \
    !defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA)
    if (ompt_multiplex_own_callbacks.ompt_callback_implicit_task) {
      ompt_multiplex_own_callbacks.ompt_callback_implicit_task(
          endpoint, ompt_multiplex_get_own_parallel_data(parallel_data),
          ompt_multiplex_get_own_task_data(task_data), team_size, thread_num,
          flags);
    }
#endif

    if (ompt_multiplex_client_callbacks.ompt_callback_implicit_task) {
      ompt_multiplex_client_callbacks.ompt_callback_implicit_task(
          endpoint, ompt_multiplex_get_client_parallel_data(parallel_data),
          ompt_multiplex_get_client_task_data(task_data), team_size, thread_num,
          flags);
    }

#if defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA) &&                     \
    !defined(OMPT_MULTIPLEX_CUSTOM_DELETE_TASK_DATA)
    if (ompt_multiplex_own_callbacks.ompt_callback_implicit_task) {
      ompt_multiplex_own_callbacks.ompt_callback_implicit_task(
          endpoint, ompt_multiplex_get_own_parallel_data(parallel_data),
          ompt_multiplex_get_own_task_data(task_data), team_size, thread_num,
          flags);
    }
#endif

#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA
    ompt_multiplex_free_data_pair(task_data);
#endif

#if defined(OMPT_MULTIPLEX_CUSTOM_DELETE_PARALLEL_DATA)
    if (flags & ompt_task_initial)
      OMPT_MULTIPLEX_CUSTOM_DELETE_PARALLEL_DATA(parallel_data);
#endif
#if defined(OMPT_MULTIPLEX_CUSTOM_DELETE_TASK_DATA)
    OMPT_MULTIPLEX_CUSTOM_DELETE_TASK_DATA(task_data);
#endif
  }
}

static void ompt_multiplex_callback_lock_init(ompt_mutex_t kind,
                                              unsigned int hint,
                                              unsigned int impl,
                                              ompt_wait_id_t wait_id,
                                              const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_lock_init) {
    ompt_multiplex_own_callbacks.ompt_callback_lock_init(kind, hint, impl,
                                                         wait_id, codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_lock_init) {
    ompt_multiplex_client_callbacks.ompt_callback_lock_init(
        kind, hint, impl, wait_id, codeptr_ra);
  }
}

static void ompt_multiplex_callback_lock_destroy(ompt_mutex_t kind,
                                                 ompt_wait_id_t wait_id,
                                                 const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_lock_destroy) {
    ompt_multiplex_own_callbacks.ompt_callback_lock_destroy(kind, wait_id,
                                                            codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_lock_destroy) {
    ompt_multiplex_client_callbacks.ompt_callback_lock_destroy(kind, wait_id,
                                                               codeptr_ra);
  }
}

static void ompt_multiplex_callback_work(ompt_work_t wstype,
                                         ompt_scope_endpoint_t endpoint,
                                         ompt_data_t *parallel_data,
                                         ompt_data_t *task_data, uint64_t count,
                                         const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_work) {
    ompt_multiplex_own_callbacks.ompt_callback_work(
        wstype, endpoint, ompt_multiplex_get_own_parallel_data(parallel_data),
        ompt_multiplex_get_own_task_data(task_data), count, codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_work) {
    ompt_multiplex_client_callbacks.ompt_callback_work(
        wstype, endpoint,
        ompt_multiplex_get_client_parallel_data(parallel_data),
        ompt_multiplex_get_client_task_data(task_data), count, codeptr_ra);
  }
}

static void ompt_multiplex_callback_master(ompt_scope_endpoint_t endpoint,
                                           ompt_data_t *parallel_data,
                                           ompt_data_t *task_data,
                                           const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_master) {
    ompt_multiplex_own_callbacks.ompt_callback_master(
        endpoint, ompt_multiplex_get_own_parallel_data(parallel_data),
        ompt_multiplex_get_own_task_data(task_data), codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_master) {
    ompt_multiplex_client_callbacks.ompt_callback_master(
        endpoint, ompt_multiplex_get_client_parallel_data(parallel_data),
        ompt_multiplex_get_client_task_data(task_data), codeptr_ra);
  }
}

static void ompt_multiplex_callback_parallel_begin(
    ompt_data_t *parent_task_data, const ompt_frame_t *parent_task_frame,
    ompt_data_t *parallel_data, uint32_t requested_team_size, int flag,
    const void *codeptr_ra) {
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA
  ompt_multiplex_allocate_data_pair(parallel_data);
#endif
  if (ompt_multiplex_own_callbacks.ompt_callback_parallel_begin) {
    ompt_multiplex_own_callbacks.ompt_callback_parallel_begin(
        ompt_multiplex_get_own_task_data(parent_task_data), parent_task_frame,
        ompt_multiplex_get_own_parallel_data(parallel_data),
        requested_team_size, flag, codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_parallel_begin) {
    ompt_multiplex_client_callbacks.ompt_callback_parallel_begin(
        ompt_multiplex_get_client_task_data(parent_task_data),
        parent_task_frame,
        ompt_multiplex_get_client_parallel_data(parallel_data),
        requested_team_size, flag, codeptr_ra);
  }
}

static void ompt_multiplex_callback_parallel_end(ompt_data_t *parallel_data,
                                                 ompt_data_t *task_data,
                                                 int flag,
                                                 const void *codeptr_ra) {
// defines to make sure, callbacks are called in correct order depending on
// defines set by the user
#if defined(OMPT_MULTIPLEX_CUSTOM_DELETE_PARALLEL_DATA) ||                     \
    !defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA)
  if (ompt_multiplex_own_callbacks.ompt_callback_parallel_end) {
    ompt_multiplex_own_callbacks.ompt_callback_parallel_end(
        ompt_multiplex_get_own_parallel_data(parallel_data),
        ompt_multiplex_get_own_task_data(task_data), flag, codeptr_ra);
  }
#endif

  if (ompt_multiplex_client_callbacks.ompt_callback_parallel_end) {
    ompt_multiplex_client_callbacks.ompt_callback_parallel_end(
        ompt_multiplex_get_client_parallel_data(parallel_data),
        ompt_multiplex_get_client_task_data(task_data), flag, codeptr_ra);
  }

#if defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA) &&                 \
    !defined(OMPT_MULTIPLEX_CUSTOM_DELETE_PARALLEL_DATA)
  if (ompt_multiplex_own_callbacks.ompt_callback_parallel_end) {
    ompt_multiplex_own_callbacks.ompt_callback_parallel_end(
        ompt_multiplex_get_own_parallel_data(parallel_data),
        ompt_multiplex_get_own_task_data(task_data), flag, codeptr_ra);
  }
#endif

#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA
  ompt_multiplex_free_data_pair(parallel_data);
#endif

#if defined(OMPT_MULTIPLEX_CUSTOM_DELETE_PARALLEL_DATA)
  OMPT_MULTIPLEX_CUSTOM_DELETE_PARALLEL_DATA(parallel_data);
#endif
}

static void ompt_multiplex_callback_task_create(
    ompt_data_t *parent_task_data, const ompt_frame_t *parent_frame,
    ompt_data_t *new_task_data, int type, int has_dependences,
    const void *codeptr_ra) {
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA
  ompt_multiplex_allocate_data_pair(new_task_data);
#endif

#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA
  if (type & ompt_task_initial) {
    ompt_data_t *parallel_data;
    ompt_multiplex_get_parallel_info(0, &parallel_data, NULL);
    ompt_multiplex_allocate_data_pair(parallel_data);
  }
#endif

  if (ompt_multiplex_own_callbacks.ompt_callback_task_create) {
    ompt_multiplex_own_callbacks.ompt_callback_task_create(
        ompt_multiplex_get_own_task_data(parent_task_data), parent_frame,
        ompt_multiplex_get_own_task_data(new_task_data), type, has_dependences,
        codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_task_create) {
    ompt_multiplex_client_callbacks.ompt_callback_task_create(
        ompt_multiplex_get_client_task_data(parent_task_data), parent_frame,
        ompt_multiplex_get_client_task_data(new_task_data), type,
        has_dependences, codeptr_ra);
  }
}

static void
ompt_multiplex_callback_task_schedule(ompt_data_t *first_task_data,
                                      ompt_task_status_t prior_task_status,
                                      ompt_data_t *second_task_data) {
  if (prior_task_status != ompt_task_complete) {
    if (ompt_multiplex_own_callbacks.ompt_callback_task_schedule) {
      ompt_multiplex_own_callbacks.ompt_callback_task_schedule(
          ompt_multiplex_get_own_task_data(first_task_data), prior_task_status,
          ompt_multiplex_get_own_task_data(second_task_data));
    }
    if (ompt_multiplex_client_callbacks.ompt_callback_task_schedule) {
      ompt_multiplex_client_callbacks.ompt_callback_task_schedule(
          ompt_multiplex_get_client_task_data(first_task_data),
          prior_task_status,
          ompt_multiplex_get_client_task_data(second_task_data));
    }
  } else {
// defines to make sure, callbacks are called in correct order depending on
// defines set by the user
#if defined(OMPT_MULTIPLEX_CUSTOM_DELETE_TASK_DATA) ||                         \
    !defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA)
    if (ompt_multiplex_own_callbacks.ompt_callback_task_schedule) {
      ompt_multiplex_own_callbacks.ompt_callback_task_schedule(
          ompt_multiplex_get_own_task_data(first_task_data), prior_task_status,
          ompt_multiplex_get_own_task_data(second_task_data));
    }
#endif

    if (ompt_multiplex_client_callbacks.ompt_callback_task_schedule) {
      ompt_multiplex_client_callbacks.ompt_callback_task_schedule(
          ompt_multiplex_get_client_task_data(first_task_data),
          prior_task_status,
          ompt_multiplex_get_client_task_data(second_task_data));
    }

#if defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA) &&                     \
    !defined(OMPT_MULTIPLEX_CUSTOM_DELETE_TASK_DATA)
    if (ompt_multiplex_own_callbacks.ompt_callback_task_schedule) {
      ompt_multiplex_own_callbacks.ompt_callback_task_schedule(
          ompt_multiplex_get_own_task_data(first_task_data), prior_task_status,
          ompt_multiplex_get_own_task_data(second_task_data));
    }
#endif

#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA
    ompt_multiplex_free_data_pair(first_task_data);
#endif

#if defined(OMPT_MULTIPLEX_CUSTOM_DELETE_TASK_DATA)
    OMPT_MULTIPLEX_CUSTOM_DELETE_TASK_DATA(first_task_data);
#endif
  }
}

static void ompt_multiplex_callback_dependences(ompt_data_t *task_data,
                                                const ompt_dependence_t *deps,
                                                int ndeps) {
  if (ompt_multiplex_own_callbacks.ompt_callback_dependences) {
    ompt_multiplex_own_callbacks.ompt_callback_dependences(
        ompt_multiplex_get_own_task_data(task_data), deps, ndeps);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_dependences) {
    ompt_multiplex_client_callbacks.ompt_callback_dependences(
        ompt_multiplex_get_client_task_data(task_data), deps, ndeps);
  }
}

static void
ompt_multiplex_callback_task_dependence(ompt_data_t *first_task_data,
                                        ompt_data_t *second_task_data) {
  if (ompt_multiplex_own_callbacks.ompt_callback_task_dependence) {
    ompt_multiplex_own_callbacks.ompt_callback_task_dependence(
        ompt_multiplex_get_own_task_data(first_task_data),
        ompt_multiplex_get_own_task_data(second_task_data));
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_task_dependence) {
    ompt_multiplex_client_callbacks.ompt_callback_task_dependence(
        ompt_multiplex_get_client_task_data(first_task_data),
        ompt_multiplex_get_client_task_data(second_task_data));
  }
}

static void ompt_multiplex_callback_thread_begin(ompt_thread_t thread_type,
                                                 ompt_data_t *thread_data) {
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA
  ompt_multiplex_allocate_data_pair(thread_data);
#endif
  if (ompt_multiplex_own_callbacks.ompt_callback_thread_begin) {
    ompt_multiplex_own_callbacks.ompt_callback_thread_begin(
        thread_type, ompt_multiplex_get_own_thread_data(thread_data));
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_thread_begin) {
    ompt_multiplex_client_callbacks.ompt_callback_thread_begin(
        thread_type, ompt_multiplex_get_client_thread_data(thread_data));
  }
}

static void ompt_multiplex_callback_thread_end(ompt_data_t *thread_data) {
// defines to make sure, callbacks are called in correct order depending on
// defines set by the user
#if defined(OMPT_MULTIPLEX_CUSTOM_DELETE_THREAD_DATA) ||                       \
    !defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA)
  if (ompt_multiplex_own_callbacks.ompt_callback_thread_end) {
    ompt_multiplex_own_callbacks.ompt_callback_thread_end(
        ompt_multiplex_get_own_thread_data(thread_data));
  }
#endif

  if (ompt_multiplex_client_callbacks.ompt_callback_thread_end) {
    ompt_multiplex_client_callbacks.ompt_callback_thread_end(
        ompt_multiplex_get_client_thread_data(thread_data));
  }

#if defined(OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA) &&                   \
    !defined(OMPT_MULTIPLEX_CUSTOM_DELETE_THREAD_DATA)
  if (ompt_multiplex_own_callbacks.ompt_callback_thread_end) {
    ompt_multiplex_own_callbacks.ompt_callback_thread_end(
        ompt_multiplex_get_own_thread_data(thread_data));
  }
#endif

#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA
  ompt_multiplex_free_data_pair(thread_data);
#endif

#if defined(OMPT_MULTIPLEX_CUSTOM_DELETE_THREAD_DATA)
  OMPT_MULTIPLEX_CUSTOM_DELETE_THREAD_DATA(thread_data);
#endif
}

static int ompt_multiplex_callback_control_tool(uint64_t command,
                                                uint64_t modifier, void *arg,
                                                const void *codeptr_ra) {
  int ownRet = 0, clientRet = 0;
  if (ompt_multiplex_own_callbacks.ompt_callback_control_tool) {
    ownRet = ompt_multiplex_own_callbacks.ompt_callback_control_tool(
        command, modifier, arg, codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_control_tool) {
    clientRet = ompt_multiplex_client_callbacks.ompt_callback_control_tool(
        command, modifier, arg, codeptr_ra);
  }
  return ownRet < clientRet ? ownRet : clientRet;
}

static void ompt_multiplex_callback_target(
    ompt_target_t kind, ompt_scope_endpoint_t endpoint, int device_num,
    ompt_data_t *task_data, ompt_id_t target_id, const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_target) {
    ompt_multiplex_own_callbacks.ompt_callback_target(
        kind, endpoint, device_num, ompt_multiplex_get_own_task_data(task_data),
        target_id, codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_target) {
    ompt_multiplex_client_callbacks.ompt_callback_target(
        kind, endpoint, device_num,
        ompt_multiplex_get_client_task_data(task_data), target_id, codeptr_ra);
  }
}

static void ompt_multiplex_callback_target_data_op(
    ompt_id_t target_id, ompt_id_t host_op_id, ompt_target_data_op_t optype,
    void *src_addr, int src_device_num, void *dest_addr, int dest_device_num,
    size_t bytes, const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_target_data_op) {
    ompt_multiplex_own_callbacks.ompt_callback_target_data_op(
        target_id, host_op_id, optype, src_addr, src_device_num, dest_addr,
        dest_device_num, bytes, codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_target_data_op) {
    ompt_multiplex_client_callbacks.ompt_callback_target_data_op(
        target_id, host_op_id, optype, src_addr, src_device_num, dest_addr,
        dest_device_num, bytes, codeptr_ra);
  }
}

static void
ompt_multiplex_callback_target_submit(ompt_id_t target_id, ompt_id_t host_op_id,
                                      unsigned int requested_num_teams) {
  if (ompt_multiplex_own_callbacks.ompt_callback_target_submit) {
    ompt_multiplex_own_callbacks.ompt_callback_target_submit(
        target_id, host_op_id, requested_num_teams);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_target_submit) {
    ompt_multiplex_client_callbacks.ompt_callback_target_submit(
        target_id, host_op_id, requested_num_teams);
  }
}

static void ompt_multiplex_callback_device_initialize(
    int device_num, const char *type, ompt_device_t *device,
    ompt_function_lookup_t lookup, const char *documentation) {
  if (ompt_multiplex_own_callbacks.ompt_callback_device_initialize) {
    ompt_multiplex_own_callbacks.ompt_callback_device_initialize(
        device_num, type, device, lookup, documentation);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_device_initialize) {
    ompt_multiplex_client_callbacks.ompt_callback_device_initialize(
        device_num, type, device, lookup, documentation);
  }
}

static void ompt_multiplex_callback_device_finalize(int device_num) {
  if (ompt_multiplex_own_callbacks.ompt_callback_device_finalize) {
    ompt_multiplex_own_callbacks.ompt_callback_device_finalize(device_num);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_device_finalize) {
    ompt_multiplex_client_callbacks.ompt_callback_device_finalize(device_num);
  }
}

static void
ompt_multiplex_callback_device_load(int device_num, const char *filename,
                                    int64_t offset_in_file, void *vma_in_file,
                                    size_t bytes, void *host_addr,
                                    void *device_addr, uint64_t module_id) {
  if (ompt_multiplex_own_callbacks.ompt_callback_device_load) {
    ompt_multiplex_own_callbacks.ompt_callback_device_load(
        device_num, filename, offset_in_file, vma_in_file, bytes, host_addr,
        device_addr, module_id);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_device_load) {
    ompt_multiplex_client_callbacks.ompt_callback_device_load(
        device_num, filename, offset_in_file, vma_in_file, bytes, host_addr,
        device_addr, module_id);
  }
}

static void ompt_multiplex_callback_device_unload(int device_num,
                                                  uint64_t module_id) {
  if (ompt_multiplex_own_callbacks.ompt_callback_device_unload) {
    ompt_multiplex_own_callbacks.ompt_callback_device_unload(device_num,
                                                             module_id);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_device_unload) {
    ompt_multiplex_client_callbacks.ompt_callback_device_unload(device_num,
                                                                module_id);
  }
}

static void
ompt_multiplex_callback_target_map(ompt_id_t target_id, unsigned int nitems,
                                   void **host_addr, void **device_addr,
                                   size_t *bytes, unsigned int *mapping_flags,
                                   const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_target_map) {
    ompt_multiplex_own_callbacks.ompt_callback_target_map(
        target_id, nitems, host_addr, device_addr, bytes, mapping_flags,
        codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_target_map) {
    ompt_multiplex_client_callbacks.ompt_callback_target_map(
        target_id, nitems, host_addr, device_addr, bytes, mapping_flags,
        codeptr_ra);
  }
}

static void ompt_multiplex_callback_reduction(ompt_sync_region_t kind,
                                              ompt_scope_endpoint_t endpoint,
                                              ompt_data_t *parallel_data,
                                              ompt_data_t *task_data,
                                              const void *codeptr_ra) {
  if (ompt_multiplex_own_callbacks.ompt_callback_reduction) {
    ompt_multiplex_own_callbacks.ompt_callback_reduction(
        kind, endpoint, ompt_multiplex_get_own_parallel_data(parallel_data),
        ompt_multiplex_get_own_task_data(task_data), codeptr_ra);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_reduction) {
    ompt_multiplex_client_callbacks.ompt_callback_reduction(
        kind, endpoint, ompt_multiplex_get_client_parallel_data(parallel_data),
        ompt_multiplex_get_client_task_data(task_data), codeptr_ra);
  }
}

static void ompt_multiplex_callback_dispatch(ompt_data_t *parallel_data,
                                             ompt_data_t *task_data,
                                             ompt_dispatch_t kind,
                                             ompt_data_t instance) {
  if (ompt_multiplex_own_callbacks.ompt_callback_dispatch) {
    ompt_multiplex_own_callbacks.ompt_callback_dispatch(
        ompt_multiplex_get_own_parallel_data(parallel_data),
        ompt_multiplex_get_own_task_data(task_data), kind, instance);
  }
  if (ompt_multiplex_client_callbacks.ompt_callback_dispatch) {
    ompt_multiplex_client_callbacks.ompt_callback_dispatch(
        ompt_multiplex_get_client_parallel_data(parallel_data),
        ompt_multiplex_get_client_task_data(task_data), kind, instance);
  }
}

// runtime entry functions

int ompt_multiplex_own_get_task_info(int ancestor_level, int *type,
                                     ompt_data_t **task_data,
                                     ompt_frame_t **task_frame,
                                     ompt_data_t **parallel_data,
                                     int *thread_num) {
  int ret = ompt_multiplex_get_task_info(ancestor_level, type, task_data,
                                         task_frame, parallel_data, thread_num);

#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA
  if (task_data)
    *task_data = ompt_multiplex_get_own_ompt_data(*task_data);
#endif
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA
  if (parallel_data)
    *parallel_data = ompt_multiplex_get_own_ompt_data(*parallel_data);
#endif
  return ret;
}

int ompt_multiplex_client_get_task_info(int ancestor_level, int *type,
                                        ompt_data_t **task_data,
                                        ompt_frame_t **task_frame,
                                        ompt_data_t **parallel_data,
                                        int *thread_num) {
  int ret = ompt_multiplex_get_task_info(ancestor_level, type, task_data,
                                         task_frame, parallel_data, thread_num);

  if (task_data)
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA
    *task_data = ompt_multiplex_get_client_ompt_data(*task_data);
#else
    *task_data = OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_TASK_DATA(*task_data);
#endif

  if (parallel_data)
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA
    *parallel_data = ompt_multiplex_get_client_ompt_data(*parallel_data);
#else
    *parallel_data =
        OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA(*parallel_data);
#endif
  return ret;
}

ompt_data_t *ompt_multiplex_own_get_thread_data() {
  ompt_data_t *ret;
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA
  ret = ompt_multiplex_get_own_ompt_data(ompt_multiplex_get_thread_data());
#else
  ret = ompt_multiplex_get_thread_data();
#endif
  return ret;
}

ompt_data_t *ompt_multiplex_client_get_thread_data() {
  ompt_data_t *ret;
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA
  ret = ompt_multiplex_get_client_ompt_data(ompt_multiplex_get_thread_data());
#else
  ret = OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_THREAD_DATA(
      ompt_multiplex_get_thread_data());
#endif
  return ret;
}

int ompt_multiplex_own_get_parallel_info(int ancestor_level,
                                         ompt_data_t **parallel_data,
                                         int *team_size) {
  int ret = ompt_multiplex_get_parallel_info(ancestor_level, parallel_data,
                                             team_size);
  if (parallel_data)
    *parallel_data = ompt_multiplex_get_own_parallel_data(*parallel_data);
  return ret;
}

int ompt_multiplex_client_get_parallel_info(int ancestor_level,
                                            ompt_data_t **parallel_data,
                                            int *team_size) {
  int ret = ompt_multiplex_get_parallel_info(ancestor_level, parallel_data,
                                             team_size);
  if (parallel_data)
#ifndef OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA
    *parallel_data = ompt_multiplex_get_client_ompt_data(*parallel_data);
#else
    *parallel_data =
        OMPT_MULTIPLEX_CUSTOM_GET_CLIENT_PARALLEL_DATA(*parallel_data);
#endif
  return ret;
}

OMPT_API_ROUTINE int ompt_multiplex_own_set_callback(ompt_callbacks_t which,
                                                     ompt_callback_t callback) {
  switch (which) {

#define ompt_event_macro(event_name, callback_type, event_id)                  \
  case ompt_##event_name:                                                      \
    ompt_multiplex_own_callbacks.ompt_##event_name = (callback_type)callback;  \
    if (ompt_multiplex_implementation_status.ompt_##event_name == -1)          \
      return ompt_multiplex_implementation_status.ompt_##event_name =          \
                 ompt_multiplex_set_callback(                                  \
                     ompt_##event_name,                                        \
                     (ompt_callback_t)&ompt_multiplex_##event_name);           \
    else                                                                       \
      return ompt_multiplex_implementation_status.ompt_##event_name

    OMPT_LOAD_CLIENT_FOREACH_OMPT_EVENT(ompt_event_macro)

#undef ompt_event_macro

  default:
    return ompt_set_error;
  }
}

OMPT_API_ROUTINE int
ompt_multiplex_client_set_callback(ompt_callbacks_t which,
                                   ompt_callback_t callback) {
  switch (which) {

#define ompt_event_macro(event_name, callback_type, event_id)                  \
  case ompt_##event_name:                                                      \
    ompt_multiplex_client_callbacks.ompt_##event_name =                        \
        (callback_type)callback;                                               \
    if (ompt_multiplex_implementation_status.ompt_##event_name == -1)          \
      return ompt_multiplex_implementation_status.ompt_##event_name =          \
                 ompt_multiplex_set_callback(                                  \
                     ompt_##event_name,                                        \
                     (ompt_callback_t)&ompt_multiplex_##event_name);           \
    else                                                                       \
      return ompt_multiplex_implementation_status.ompt_##event_name

    OMPT_LOAD_CLIENT_FOREACH_OMPT_EVENT(ompt_event_macro)

#undef ompt_event_macro

  default:
    return ompt_set_error;
  }
}

ompt_interface_fn_t ompt_multiplex_own_lookup(const char *name) {
  if (!strcmp(name, "ompt_set_callback"))
    return (ompt_interface_fn_t)&ompt_multiplex_own_set_callback;
  else if (!strcmp(name, "ompt_get_task_info"))
    return (ompt_interface_fn_t)&ompt_multiplex_own_get_task_info;
  else if (!strcmp(name, "ompt_get_thread_data"))
    return (ompt_interface_fn_t)&ompt_multiplex_own_get_thread_data;
  else if (!strcmp(name, "ompt_get_parallel_info"))
    return (ompt_interface_fn_t)&ompt_multiplex_own_get_parallel_info;
  else
    return ompt_multiplex_lookup_function(name);
}

ompt_interface_fn_t ompt_multiplex_client_lookup(const char *name) {
  if (!strcmp(name, "ompt_set_callback"))
    return (ompt_interface_fn_t)&ompt_multiplex_client_set_callback;
  else if (!strcmp(name, "ompt_get_task_info"))
    return (ompt_interface_fn_t)&ompt_multiplex_client_get_task_info;
  else if (!strcmp(name, "ompt_get_thread_data"))
    return (ompt_interface_fn_t)&ompt_multiplex_client_get_thread_data;
  else if (!strcmp(name, "ompt_get_parallel_info"))
    return (ompt_interface_fn_t)&ompt_multiplex_client_get_parallel_info;
  else
    return ompt_multiplex_lookup_function(name);
}

int ompt_multiplex_initialize(ompt_function_lookup_t lookup,
                              int initial_device_num, ompt_data_t *data) {
  ompt_multiplex_lookup_function = lookup;
  ompt_multiplex_set_callback =
      (ompt_set_callback_t)lookup("ompt_set_callback");
  ompt_multiplex_get_task_info =
      (ompt_get_task_info_t)lookup("ompt_get_task_info");
  ompt_multiplex_get_thread_data =
      (ompt_get_thread_data_t)lookup("ompt_get_thread_data");
  ompt_multiplex_get_parallel_info =
      (ompt_get_parallel_info_t)lookup("ompt_get_parallel_info");

  // initialize ompt_multiplex_implementation_status
#define ompt_event_macro(event_name, callback_type, event_id)                  \
  ompt_multiplex_implementation_status.ompt_##event_name = -1

  OMPT_LOAD_CLIENT_FOREACH_OMPT_EVENT(ompt_event_macro)

#undef ompt_event_macro

  int ownRet = ompt_multiplex_own_fns->initialize(
      ompt_multiplex_own_lookup, initial_device_num,
      &(ompt_multiplex_own_fns->tool_data));
  int clientRet = 0;
  if (ompt_multiplex_client_fns)
    clientRet = ompt_multiplex_client_fns->initialize(
        ompt_multiplex_client_lookup, initial_device_num,
        &(ompt_multiplex_client_fns->tool_data));

  return ownRet > clientRet ? ownRet : clientRet;
}

void ompt_multiplex_finalize(ompt_data_t *fns) {
  if (ompt_multiplex_client_fns)
    ompt_multiplex_client_fns->finalize(
        &(ompt_multiplex_client_fns->tool_data));
  ompt_multiplex_own_fns->finalize(&(ompt_multiplex_own_fns->tool_data));
}

#ifdef __cplusplus
extern "C" {
#endif
ompt_start_tool_result_t *
ompt_multiplex_own_start_tool(unsigned int omp_version,
                              const char *runtime_version);

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  // try loading client tool
  ompt_multiplex_client_fns = NULL;
  ompt_start_tool_result_t *(*client_start_tool)(unsigned int, const char *) =
      NULL;

  const char *tool_libs = getenv(CLIENT_TOOL_LIBRARIES_VAR);
  if (tool_libs) {
    // copy environement variable
    char *tool_libs_buffer = strdup(tool_libs);
    if (!tool_libs_buffer) {
      printf("strdup Error (%i)\n", errno);
      exit(-1);
    }

    int progress = 0;
    while (progress < strlen(tool_libs)) {
      int tmp_progress = progress;
      while (tmp_progress < strlen(tool_libs) &&
             tool_libs_buffer[tmp_progress] != ':')
        tmp_progress++;
      if (tmp_progress < strlen(tool_libs))
        tool_libs_buffer[tmp_progress] = 0;
      void *h = dlopen(tool_libs_buffer + progress, RTLD_LAZY);
      if (h) {
        client_start_tool =
            (ompt_start_tool_result_t * (*)(unsigned int, const char *))
                dlsym(h, "ompt_start_tool");
        if (client_start_tool &&
            (ompt_multiplex_client_fns =
                 (*client_start_tool)(omp_version, runtime_version))) {
          break;
        }
      } else {
        printf("Loading %s from %s failed with: %s\n",
               tool_libs_buffer + progress, CLIENT_TOOL_LIBRARIES_VAR,
               dlerror());
      }
      progress = tmp_progress + 1;
    }
    free(tool_libs_buffer);
  }
  // load own tool
  ompt_multiplex_own_fns =
      ompt_multiplex_own_start_tool(omp_version, runtime_version);

  // return multiplexed versions
  static ompt_start_tool_result_t ompt_start_tool_result = {
      &ompt_multiplex_initialize, &ompt_multiplex_finalize, {0}};
  return &ompt_start_tool_result;
}
#ifdef __cplusplus
}
#endif

// We rename the ompt_start_tool function of the OMPT tool and call the
// renamed function from the ompt_start_tool function defined above.
#define ompt_start_tool ompt_multiplex_own_start_tool

#endif /* OMPT_MULTIPLEX_H */
