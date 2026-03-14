/*
 * ompd-specific.h -- OpenMP debug support
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp.h"
#include "omp-tools.h"
#include <stdint.h>

#ifndef __OMPD_SPECIFIC_H__
#define __OMPD_SPECIFIC_H__

#if OMPD_SUPPORT

void ompd_init();

#ifdef __cplusplus
extern "C" {
#endif
extern char *ompd_env_block;
extern ompd_size_t ompd_env_block_size;
extern char *__kmp_tool_verbose_init;
#ifdef __cplusplus
} /* extern "C" */
#endif

extern uint64_t ompd_state;
#define OMPD_ENABLE_BP 0x1

#define OMPD_FOREACH_ACCESS(OMPD_ACCESS)                                       \
  OMPD_ACCESS(kmp_base_info_t, th_current_task)                                \
  OMPD_ACCESS(kmp_base_info_t, th_team)                                        \
  OMPD_ACCESS(kmp_base_info_t, th_info)                                        \
  OMPD_ACCESS(kmp_base_info_t, ompt_thread_info)                               \
                                                                               \
  OMPD_ACCESS(kmp_base_root_t, r_in_parallel)                                  \
                                                                               \
  OMPD_ACCESS(kmp_base_team_t, ompt_team_info)                                 \
  OMPD_ACCESS(kmp_base_team_t, ompt_serialized_team_info)                      \
  OMPD_ACCESS(kmp_base_team_t, t_active_level)                                 \
  OMPD_ACCESS(kmp_base_team_t, t_implicit_task_taskdata)                       \
  OMPD_ACCESS(kmp_base_team_t, t_master_tid)                                   \
  OMPD_ACCESS(kmp_base_team_t, t_nproc)                                        \
  OMPD_ACCESS(kmp_base_team_t, t_level)                                        \
  OMPD_ACCESS(kmp_base_team_t, t_parent)                                       \
  OMPD_ACCESS(kmp_base_team_t, t_pkfn)                                         \
  OMPD_ACCESS(kmp_base_team_t, t_threads)                                      \
                                                                               \
  OMPD_ACCESS(kmp_desc_t, ds)                                                  \
                                                                               \
  OMPD_ACCESS(kmp_desc_base_t, ds_thread)                                      \
  OMPD_ACCESS(kmp_desc_base_t, ds_tid)                                         \
                                                                               \
  OMPD_ACCESS(kmp_info_t, th)                                                  \
                                                                               \
  OMPD_ACCESS(kmp_r_sched_t, r_sched_type)                                     \
  OMPD_ACCESS(kmp_r_sched_t, chunk)                                            \
                                                                               \
  OMPD_ACCESS(kmp_root_t, r)                                                   \
                                                                               \
  OMPD_ACCESS(kmp_internal_control_t, dynamic)                                 \
  OMPD_ACCESS(kmp_internal_control_t, max_active_levels)                       \
  OMPD_ACCESS(kmp_internal_control_t, nproc)                                   \
  OMPD_ACCESS(kmp_internal_control_t, proc_bind)                               \
  OMPD_ACCESS(kmp_internal_control_t, sched)                                   \
  OMPD_ACCESS(kmp_internal_control_t, default_device)                          \
  OMPD_ACCESS(kmp_internal_control_t, thread_limit)                            \
                                                                               \
  OMPD_ACCESS(kmp_taskdata_t, ompt_task_info)                                  \
  OMPD_ACCESS(kmp_taskdata_t, td_flags)                                        \
  OMPD_ACCESS(kmp_taskdata_t, td_icvs)                                         \
  OMPD_ACCESS(kmp_taskdata_t, td_parent)                                       \
  OMPD_ACCESS(kmp_taskdata_t, td_team)                                         \
                                                                               \
  OMPD_ACCESS(kmp_task_t, routine)                                             \
                                                                               \
  OMPD_ACCESS(kmp_team_p, t)                                                   \
                                                                               \
  OMPD_ACCESS(kmp_nested_nthreads_t, used)                                     \
  OMPD_ACCESS(kmp_nested_nthreads_t, nth)                                      \
                                                                               \
  OMPD_ACCESS(kmp_nested_proc_bind_t, used)                                    \
  OMPD_ACCESS(kmp_nested_proc_bind_t, bind_types)                              \
                                                                               \
  OMPD_ACCESS(ompt_task_info_t, frame)                                         \
  OMPD_ACCESS(ompt_task_info_t, scheduling_parent)                             \
  OMPD_ACCESS(ompt_task_info_t, task_data)                                     \
                                                                               \
  OMPD_ACCESS(ompt_team_info_t, parallel_data)                                 \
                                                                               \
  OMPD_ACCESS(ompt_thread_info_t, state)                                       \
  OMPD_ACCESS(ompt_thread_info_t, wait_id)                                     \
  OMPD_ACCESS(ompt_thread_info_t, thread_data)                                 \
                                                                               \
  OMPD_ACCESS(ompt_data_t, value)                                              \
  OMPD_ACCESS(ompt_data_t, ptr)                                                \
                                                                               \
  OMPD_ACCESS(ompt_frame_t, exit_frame)                                        \
  OMPD_ACCESS(ompt_frame_t, enter_frame)                                       \
                                                                               \
  OMPD_ACCESS(ompt_lw_taskteam_t, parent)                                      \
  OMPD_ACCESS(ompt_lw_taskteam_t, ompt_team_info)                              \
  OMPD_ACCESS(ompt_lw_taskteam_t, ompt_task_info)

#define OMPD_FOREACH_BITFIELD(OMPD_BITFIELD)                                   \
  OMPD_BITFIELD(kmp_tasking_flags_t, final)                                    \
  OMPD_BITFIELD(kmp_tasking_flags_t, tiedness)                                 \
  OMPD_BITFIELD(kmp_tasking_flags_t, tasktype)                                 \
  OMPD_BITFIELD(kmp_tasking_flags_t, task_serial)                              \
  OMPD_BITFIELD(kmp_tasking_flags_t, tasking_ser)                              \
  OMPD_BITFIELD(kmp_tasking_flags_t, team_serial)                              \
  OMPD_BITFIELD(kmp_tasking_flags_t, started)                                  \
  OMPD_BITFIELD(kmp_tasking_flags_t, executing)                                \
  OMPD_BITFIELD(kmp_tasking_flags_t, complete)                                 \
  OMPD_BITFIELD(kmp_tasking_flags_t, freed)                                    \
  OMPD_BITFIELD(kmp_tasking_flags_t, native)

#define OMPD_FOREACH_SIZEOF(OMPD_SIZEOF)                                       \
  OMPD_SIZEOF(kmp_info_t)                                                      \
  OMPD_SIZEOF(kmp_taskdata_t)                                                  \
  OMPD_SIZEOF(kmp_task_t)                                                      \
  OMPD_SIZEOF(kmp_tasking_flags_t)                                             \
  OMPD_SIZEOF(kmp_thread_t)                                                    \
  OMPD_SIZEOF(ompt_data_t)                                                     \
  OMPD_SIZEOF(ompt_id_t)                                                       \
  OMPD_SIZEOF(__kmp_avail_proc)                                                \
  OMPD_SIZEOF(__kmp_max_nth)                                                   \
  OMPD_SIZEOF(__kmp_stksize)                                                   \
  OMPD_SIZEOF(__kmp_omp_cancellation)                                          \
  OMPD_SIZEOF(__kmp_max_task_priority)                                         \
  OMPD_SIZEOF(__kmp_display_affinity)                                          \
  OMPD_SIZEOF(__kmp_affinity_format)                                           \
  OMPD_SIZEOF(__kmp_tool_libraries)                                            \
  OMPD_SIZEOF(__kmp_tool_verbose_init)                                         \
  OMPD_SIZEOF(__kmp_tool)                                                      \
  OMPD_SIZEOF(ompd_state)                                                      \
  OMPD_SIZEOF(kmp_nested_nthreads_t)                                           \
  OMPD_SIZEOF(__kmp_nested_nth)                                                \
  OMPD_SIZEOF(kmp_nested_proc_bind_t)                                          \
  OMPD_SIZEOF(__kmp_nested_proc_bind)                                          \
  OMPD_SIZEOF(int)                                                             \
  OMPD_SIZEOF(char)                                                            \
  OMPD_SIZEOF(__kmp_gtid)                                                      \
  OMPD_SIZEOF(__kmp_nth)

#endif /* OMPD_SUPPORT */
#endif
