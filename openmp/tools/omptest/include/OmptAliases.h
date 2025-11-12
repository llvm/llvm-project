//===- OmptAliases.h - Shorthand aliases for OMPT enum values ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines shorthand aliases for OMPT enum values, providing improved
/// ease-of-use and readability.
///
//===----------------------------------------------------------------------===//

#ifndef OPENMP_TOOLS_OMPTEST_INCLUDE_OMPTALIASES_H
#define OPENMP_TOOLS_OMPTEST_INCLUDE_OMPTALIASES_H

#include <omp-tools.h>

/// Aliases for enum: ompt_scope_endpoint_t
constexpr ompt_scope_endpoint_t BEGIN = ompt_scope_begin;
constexpr ompt_scope_endpoint_t END = ompt_scope_end;
constexpr ompt_scope_endpoint_t BEGINEND = ompt_scope_beginend;

/// Aliases for enum: ompt_target_t
constexpr ompt_target_t TARGET = ompt_target;
constexpr ompt_target_t ENTER_DATA = ompt_target_enter_data;
constexpr ompt_target_t EXIT_DATA = ompt_target_exit_data;
constexpr ompt_target_t UPDATE = ompt_target_update;
constexpr ompt_target_t TARGET_NOWAIT = ompt_target_nowait;
constexpr ompt_target_t ENTER_DATA_NOWAIT = ompt_target_enter_data_nowait;
constexpr ompt_target_t EXIT_DATA_NOWAIT = ompt_target_exit_data_nowait;
constexpr ompt_target_t UPDATE_NOWAIT = ompt_target_update_nowait;

/// Aliases for enum: ompt_target_data_op_t
constexpr ompt_target_data_op_t ALLOC = ompt_target_data_alloc;
constexpr ompt_target_data_op_t H2D = ompt_target_data_transfer_to_device;
constexpr ompt_target_data_op_t D2H = ompt_target_data_transfer_from_device;
constexpr ompt_target_data_op_t DELETE = ompt_target_data_delete;
constexpr ompt_target_data_op_t ASSOCIATE = ompt_target_data_associate;
constexpr ompt_target_data_op_t DISASSOCIATE = ompt_target_data_disassociate;
constexpr ompt_target_data_op_t ALLOC_ASYNC = ompt_target_data_alloc_async;
constexpr ompt_target_data_op_t H2D_ASYNC =
    ompt_target_data_transfer_to_device_async;
constexpr ompt_target_data_op_t D2H_ASYNC =
    ompt_target_data_transfer_from_device_async;
constexpr ompt_target_data_op_t DELETE_ASYNC = ompt_target_data_delete_async;

/// Aliases for enum: ompt_callbacks_t (partial)
constexpr ompt_callbacks_t CB_TARGET = ompt_callback_target;
constexpr ompt_callbacks_t CB_DATAOP = ompt_callback_target_data_op;
constexpr ompt_callbacks_t CB_KERNEL = ompt_callback_target_submit;

/// Aliases for enum: ompt_work_t
constexpr ompt_work_t WORK_LOOP = ompt_work_loop;
constexpr ompt_work_t WORK_SECT = ompt_work_sections;
constexpr ompt_work_t WORK_EXEC = ompt_work_single_executor;
constexpr ompt_work_t WORK_SINGLE = ompt_work_single_other;
constexpr ompt_work_t WORK_SHARE = ompt_work_workshare;
constexpr ompt_work_t WORK_DIST = ompt_work_distribute;
constexpr ompt_work_t WORK_TASK = ompt_work_taskloop;
constexpr ompt_work_t WORK_SCOPE = ompt_work_scope;
constexpr ompt_work_t WORK_LOOP_STA = ompt_work_loop_static;
constexpr ompt_work_t WORK_LOOP_DYN = ompt_work_loop_dynamic;
constexpr ompt_work_t WORK_LOOP_GUI = ompt_work_loop_guided;
constexpr ompt_work_t WORK_LOOP_OTH = ompt_work_loop_other;

/// Aliases for enum: ompt_sync_region_t
constexpr ompt_sync_region_t SR_BARRIER = ompt_sync_region_barrier;
constexpr ompt_sync_region_t SR_BARRIER_IMPL =
    ompt_sync_region_barrier_implicit;
constexpr ompt_sync_region_t SR_BARRIER_EXPL =
    ompt_sync_region_barrier_explicit;
constexpr ompt_sync_region_t SR_BARRIER_IMPLEMENTATION =
    ompt_sync_region_barrier_implementation;
constexpr ompt_sync_region_t SR_TASKWAIT = ompt_sync_region_taskwait;
constexpr ompt_sync_region_t SR_TASKGROUP = ompt_sync_region_taskgroup;
constexpr ompt_sync_region_t SR_REDUCTION = ompt_sync_region_reduction;
constexpr ompt_sync_region_t SR_BARRIER_IMPL_WORKSHARE =
    ompt_sync_region_barrier_implicit_workshare;
constexpr ompt_sync_region_t SR_BARRIER_IMPL_PARALLEL =
    ompt_sync_region_barrier_implicit_parallel;
constexpr ompt_sync_region_t SR_BARRIER_TEAMS = ompt_sync_region_barrier_teams;

#endif
