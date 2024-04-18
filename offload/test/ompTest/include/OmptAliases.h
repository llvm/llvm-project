#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTALIASES_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTALIASES_H

#include <omp-tools.h>

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

#endif // include guard
