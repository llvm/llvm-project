//===----------- device.h - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for OpenMP Tool callback dispatchers
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_CALLBACK_H
#define _OMPTARGET_CALLBACK_H

#if (__PPC64__ | __arm__)
#define OMPT_GET_FRAME_ADDRESS(level) __builtin_frame_address(level)
#define OMPT_FRAME_POSITION_DEFAULT ompt_frame_cfa
#else
#define OMPT_GET_FRAME_ADDRESS(level) __builtin_frame_address(level)
#define OMPT_FRAME_POSITION_DEFAULT ompt_frame_framepointer
#endif

#define OMPT_FRAME_FLAGS (ompt_frame_runtime | OMPT_FRAME_POSITION_DEFAULT)

#define OMPT_GET_RETURN_ADDRESS(level) __builtin_return_address(level)

#include <chrono>

#include <omp-tools.h>

using HighResClk = std::chrono::high_resolution_clock;
using HighResTp = std::chrono::time_point<HighResClk>;
using DurationNs = std::chrono::nanoseconds;

class OmptInterface {
public:
  OmptInterface()
      : _enter_frame(NULL), _codeptr_ra(NULL), _state(ompt_state_idle) {}

  void ompt_state_set(void *enter_frame, void *codeptr_ra);

  void ompt_state_clear();

  // target op callbacks
  void target_data_alloc_begin(int64_t device_id, void *TgtPtrBegin,
                               size_t Size, void *codeptr);

  void target_data_alloc_end(int64_t device_id, void *TgtPtrBegin, size_t Size,
                             void *codeptr);

  void target_data_submit_begin(int64_t device_id, void *HstPtrBegin,
                                void *TgtPtrBegin, size_t Size, void *codeptr);

  void target_data_submit_end(int64_t device_id, void *HstPtrBegin,
                              void *TgtPtrBegin, size_t Size, void *codeptr);

  void target_data_delete_begin(int64_t device_id, void *TgtPtrBegin,
                                void *codeptr);

  void target_data_delete_end(int64_t device_id, void *TgtPtrBegin,
                              void *codeptr);

  void target_data_retrieve_begin(int64_t device_id, void *HstPtrBegin,
                                  void *TgtPtrBegin, size_t Size,
                                  void *codeptr);

  void target_data_retrieve_end(int64_t device_id, void *HstPtrBegin,
                                void *TgtPtrBegin, size_t Size, void *codeptr);

  void target_submit_begin(unsigned int num_teams = 1);

  void target_submit_end(unsigned int num_teams = 1);

  // target region callbacks
  void target_data_enter_begin(int64_t device_id, void *codeptr);

  void target_data_enter_end(int64_t device_id, void *codeptr);

  void target_data_exit_begin(int64_t device_id, void *codeptr);

  void target_data_exit_end(int64_t device_id, void *codeptr);

  void target_update_begin(int64_t device_id, void *codeptr);

  void target_update_end(int64_t device_id, void *codeptr);

  void target_begin(int64_t device_id, void *codeptr);

  void target_end(int64_t device_id, void *codeptr);

  uint64_t get_ns_duration_since_epoch() {
    const HighResTp time_point = HighResClk::now();
    const HighResClk::duration duration_since_epoch =
        time_point.time_since_epoch();
    return std::chrono::duration_cast<DurationNs>(duration_since_epoch).count();
  }

  ompt_record_ompt_t *target_trace_record_gen(int64_t device_id,
                                              ompt_target_t kind,
                                              ompt_scope_endpoint_t endpoint,
                                              void *code);
  ompt_record_ompt_t *
  target_submit_trace_record_gen(uint64_t start_time,
                                 unsigned int num_teams = 1);
  ompt_record_ompt_t *target_data_submit_trace_record_gen(
      int64_t device_id, ompt_target_data_op_t data_op, void *tgt_ptr,
      void *hst_ptr, size_t bytes, uint64_t start_time);

private:
  void ompt_state_set_helper(void *enter_frame, void *codeptr_ra, int flags,
                             int state);

  // begin/end target op marks
  void target_operation_begin();

  void target_operation_end();

  // begin/end target region marks
  void target_region_begin();

  void target_region_end();

  void target_region_announce(const char *name);

private:
  void *_enter_frame;
  void *_codeptr_ra;
  int _state;

  // Called by all trace generation routines
  void set_trace_record_common(ompt_record_ompt_t *data_ptr,
                               ompt_callbacks_t cbt, uint64_t start_time);
  // Type specific helpers
  void set_trace_record_target_data_op(ompt_record_target_data_op_t *rec,
                                       int64_t device_id,
                                       ompt_target_data_op_t data_op,
                                       void *src_ptr, void *dest_ptr,
                                       size_t bytes);
  void set_trace_record_target_kernel(ompt_record_target_kernel_t *rec,
                                      unsigned int num_teams);
  void set_trace_record_target(ompt_record_target_t *rec, int64_t device_id,
                               ompt_target_t kind,
                               ompt_scope_endpoint_t endpoint, void *code);
};

extern thread_local OmptInterface ompt_interface;

extern bool ompt_enabled;

#endif
