/*
 * ompt-specific.h - header of OMPT internal functions implementation
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPT_SPECIFIC_H
#define OMPT_SPECIFIC_H

#include "kmp.h"

#if OMPT_SUPPORT
/*****************************************************************************
 * forward declarations
 ****************************************************************************/

void __ompt_force_initialization();

int __ompt_set_frame_enter_internal(void *addr, int flags, int state);

void __ompt_team_assign_id(kmp_team_t *team, ompt_data_t ompt_pid);

void __ompt_thread_assign_wait_id(void *variable);

void __ompt_lw_taskteam_init(ompt_lw_taskteam_t *lwt, kmp_info_t *thr, int gtid,
                             ompt_data_t *ompt_pid, void *codeptr);

void __ompt_lw_taskteam_link(ompt_lw_taskteam_t *lwt, kmp_info_t *thr,
                             int on_heap, bool always = false);

void __ompt_lw_taskteam_unlink(kmp_info_t *thr);

ompt_team_info_t *__ompt_get_teaminfo(int depth, int *size);

ompt_data_t *__ompt_get_task_data();

ompt_data_t *__ompt_get_target_task_data();

ompt_task_info_t *__ompt_get_task_info_object(int depth);

int __ompt_get_parallel_info_internal(int ancestor_level,
                                      ompt_data_t **parallel_data,
                                      int *team_size);

int __ompt_get_task_info_internal(int ancestor_level, int *type,
                                  ompt_data_t **task_data,
                                  ompt_frame_t **task_frame,
                                  ompt_data_t **parallel_data, int *thread_num);

ompt_data_t *__ompt_get_thread_data_internal();

/*
 * Unused currently
static uint64_t __ompt_get_get_unique_id_internal();
*/

ompt_sync_region_t __ompt_get_barrier_kind(enum barrier_type, kmp_info_t *);

/*****************************************************************************
 * macros
 ****************************************************************************/

#define OMPT_CUR_TASK_INFO(thr) (&((thr)->th.th_current_task->ompt_task_info))
#define OMPT_CUR_TASK_DATA(thr)                                                \
  (&((thr)->th.th_current_task->ompt_task_info.task_data))
#define OMPT_CUR_TEAM_INFO(thr) (&((thr)->th.th_team->t.ompt_team_info))
#define OMPT_CUR_TEAM_DATA(thr)                                                \
  (&((thr)->th.th_team->t.ompt_team_info.parallel_data))

#define OMPT_HAVE_WEAK_ATTRIBUTE KMP_HAVE_WEAK_ATTRIBUTE
#define OMPT_HAVE_PSAPI KMP_HAVE_PSAPI
#define OMPT_STR_MATCH(haystack, needle) __kmp_str_match(haystack, 0, needle)

inline void *__ompt_load_return_address(int gtid) {
  kmp_info_t *thr = __kmp_threads[gtid];
  void *return_address = thr->th.ompt_thread_info.return_address;
  thr->th.ompt_thread_info.return_address = NULL;
  return return_address;
}

/*#define OMPT_STORE_RETURN_ADDRESS(gtid) \
  if (ompt_enabled.enabled && gtid >= 0 && __kmp_threads[gtid] &&              \
      !__kmp_threads[gtid]->th.ompt_thread_info.return_address)                \
  __kmp_threads[gtid]->th.ompt_thread_info.return_address =                    \
  __builtin_return_address(0)*/

#define OMPT_STORE_RETURN_ADDRESS(gtid)                                        \
  OmptReturnAddressGuard ReturnAddressGuard{gtid, __builtin_return_address(0)};
#define OMPT_LOAD_RETURN_ADDRESS(gtid) __ompt_load_return_address(gtid)
#define OMPT_LOAD_OR_GET_RETURN_ADDRESS(gtid)                                  \
  ((ompt_enabled.enabled && gtid >= 0 && __kmp_threads[gtid] &&                \
    __kmp_threads[gtid]->th.ompt_thread_info.return_address)                   \
       ? __ompt_load_return_address(gtid)                                      \
       : __builtin_return_address(0))

#define OMPT_GET_DISPATCH_CHUNK(chunk, lb, ub, incr)                           \
  do {                                                                         \
    if (incr > 0) {                                                            \
      chunk.start = static_cast<uint64_t>(lb);                                 \
      chunk.iterations = static_cast<uint64_t>(((ub) - (lb)) / (incr) + 1);    \
    } else {                                                                   \
      chunk.start = static_cast<uint64_t>(ub);                                 \
      chunk.iterations = static_cast<uint64_t>(((lb) - (ub)) / -(incr) + 1);   \
    }                                                                          \
  } while (0)

//******************************************************************************
// inline functions
//******************************************************************************

inline kmp_info_t *ompt_get_thread_gtid(int gtid) {
  return (gtid >= 0) ? __kmp_thread_from_gtid(gtid) : NULL;
}

inline kmp_info_t *ompt_get_thread() {
  int gtid = __kmp_get_gtid();
  return ompt_get_thread_gtid(gtid);
}

inline void ompt_set_thread_state(kmp_info_t *thread, ompt_state_t state) {
  if (thread)
    thread->th.ompt_thread_info.state = state;
}

inline const char *ompt_get_runtime_version() {
  return &__kmp_version_lib_ver[KMP_VERSION_MAGIC_LEN];
}

class OmptReturnAddressGuard {
private:
  bool SetAddress{false};
  int Gtid;

public:
  OmptReturnAddressGuard(int Gtid, void *ReturnAddress) : Gtid(Gtid) {
    if (ompt_enabled.enabled && Gtid >= 0 && __kmp_threads[Gtid] &&
        !__kmp_threads[Gtid]->th.ompt_thread_info.return_address) {
      SetAddress = true;
      __kmp_threads[Gtid]->th.ompt_thread_info.return_address = ReturnAddress;
    }
  }
  ~OmptReturnAddressGuard() {
    if (SetAddress)
      __kmp_threads[Gtid]->th.ompt_thread_info.return_address = NULL;
  }
};

#endif // OMPT_SUPPORT

// macros providing the OMPT callbacks for reduction clause
#if OMPT_SUPPORT && OMPT_OPTIONAL
#define OMPT_REDUCTION_DECL(this_thr, gtid)                                    \
  ompt_data_t *my_task_data = OMPT_CUR_TASK_DATA(this_thr);                    \
  ompt_data_t *my_parallel_data = OMPT_CUR_TEAM_DATA(this_thr);                \
  void *return_address = OMPT_LOAD_RETURN_ADDRESS(gtid);
#define OMPT_REDUCTION_BEGIN                                                   \
  if (ompt_enabled.enabled && ompt_enabled.ompt_callback_reduction) {          \
    ompt_callbacks.ompt_callback(ompt_callback_reduction)(                     \
        ompt_sync_region_reduction, ompt_scope_begin, my_parallel_data,        \
        my_task_data, return_address);                                         \
  }
#define OMPT_REDUCTION_END                                                     \
  if (ompt_enabled.enabled && ompt_enabled.ompt_callback_reduction) {          \
    ompt_callbacks.ompt_callback(ompt_callback_reduction)(                     \
        ompt_sync_region_reduction, ompt_scope_end, my_parallel_data,          \
        my_task_data, return_address);                                         \
  }
#else // OMPT_SUPPORT && OMPT_OPTIONAL
#define OMPT_REDUCTION_DECL(this_thr, gtid)
#define OMPT_REDUCTION_BEGIN
#define OMPT_REDUCTION_END
#endif // ! OMPT_SUPPORT && OMPT_OPTIONAL

#endif
