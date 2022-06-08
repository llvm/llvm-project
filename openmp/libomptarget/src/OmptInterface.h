//===-------- OmptInterface.h - Target independent OpenMP target RTL ------===//
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

#ifndef _OMPTARGET_OMPTINTERFACE_H
#define _OMPTARGET_OMPTINTERFACE_H

// Only provide functionality if target OMPT support is enabled
#ifdef OMPT_SUPPORT

#include "OmptCallback.h"
#include "omp-tools.h"

#define OMPT_IF_BUILT(stmt) stmt

/// Callbacks for target regions require task_data representing the
/// encountering task.
/// Callbacks for target regions and target data ops require
/// target_task_data representing the target task region.
typedef ompt_data_t *(*ompt_get_task_data_t)();
typedef ompt_data_t *(*ompt_get_target_task_data_t)();
typedef int (*ompt_set_frame_enter_t)(void *Address, int Flags, int State);

namespace llvm {
namespace omp {
namespace target {
namespace ompt {

/// Function pointers that will be used to track task_data and
/// target_task_data.
extern ompt_get_task_data_t ompt_get_task_data_fn;
extern ompt_get_target_task_data_t ompt_get_target_task_data_fn;
extern ompt_set_frame_enter_t ompt_set_frame_enter_fn;

/// Used to maintain execution state for this thread
class Interface {
public:
  /// Top-level function for invoking callback before device data allocation
  void beginTargetDataAlloc(int64_t DeviceId, void *HstPtrBegin,
                            void **TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback after device data allocation
  void endTargetDataAlloc(int64_t DeviceId, void *HstPtrBegin,
                          void **TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback before data submit
  void beginTargetDataSubmit(int64_t DeviceId, void *HstPtrBegin,
                             void *TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback after data submit
  void endTargetDataSubmit(int64_t DeviceId, void *HstPtrBegin,
                           void *TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback before device data deallocation
  void beginTargetDataDelete(int64_t DeviceId, void *TgtPtrBegin, void *Code);

  /// Top-level function for invoking callback after device data deallocation
  void endTargetDataDelete(int64_t DeviceId, void *TgtPtrBegin, void *Code);

  /// Top-level function for invoking callback before data retrieve
  void beginTargetDataRetrieve(int64_t DeviceId, void *HstPtrBegin,
                               void *TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback after data retrieve
  void endTargetDataRetrieve(int64_t DeviceId, void *HstPtrBegin,
                             void *TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback before kernel dispatch
  void beginTargetSubmit(unsigned int NumTeams = 1);

  /// Top-level function for invoking callback after kernel dispatch
  void endTargetSubmit(unsigned int NumTeams = 1);

  // Target region callbacks

  /// Top-level function for invoking callback before target enter data
  /// construct
  void beginTargetDataEnter(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback after target enter data
  /// construct
  void endTargetDataEnter(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback before target exit data
  /// construct
  void beginTargetDataExit(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback after target exit data
  /// construct
  void endTargetDataExit(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback before target update construct
  void beginTargetUpdate(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback after target update construct
  void endTargetUpdate(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback before target construct
  void beginTarget(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback after target construct
  void endTarget(int64_t DeviceId, void *Code);

  /// Setters for target region and target operation correlation ids
  void setTargetDataValue(uint64_t DataValue) { TargetData.value = DataValue; }
  void setTargetDataPtr(void *DataPtr) { TargetData.ptr = DataPtr; }
  void setHostOpId(ompt_id_t OpId) { HostOpId = OpId; }

  /// Getters for target region and target operation correlation ids
  uint64_t getTargetDataValue() { return TargetData.value; }
  void *getTargetDataPtr() { return TargetData.ptr; }
  ompt_id_t getHostOpId() { return HostOpId; }

  // ToDo: mhalk Docstrings, code style, ...
  ompt_record_ompt_t *target_trace_record_gen(int64_t device_id,
                                              ompt_target_t kind,
                                              ompt_scope_endpoint_t endpoint,
                                              void *code);

  ompt_record_ompt_t *
  target_submit_trace_record_gen(unsigned int num_teams = 1);

  ompt_record_ompt_t *target_data_submit_trace_record_gen(
      ompt_target_data_op_t data_op, void *src_addr, int64_t src_device_num,
      void *dest_addr, int64_t dest_device_num, size_t bytes);

  void ompt_state_set(void *enter_frame, void *codeptr_ra);

  void ompt_state_clear();

private:
  /// Target operations id
  ompt_id_t HostOpId = 0;

  /// Target region data
  ompt_data_t TargetData = ompt_data_none;

  /// Task data representing the encountering task
  ompt_data_t *TaskData = nullptr;

  /// Target task data representing the target task region
  ompt_data_t *TargetTaskData = nullptr;

  /// Correlation id that is incremented with target operations
  uint64_t TargetRegionOpId = 1;

  /// ToDo: mhalk ...
  void *_enter_frame;

  /// Return-Address pointer reported in a trace record
  void *_codeptr_ra;

  /// ToDo: mhalk ...
  int _state;

  /// Used for marking begin of a data operation
  void announceTargetRegion(const char *RegionName);

  /// Used for marking begin of a data operation
  void beginTargetDataOperation();

  /// Used for marking end of a data operation
  void endTargetDataOperation();

  /// Used for marking begin of a target region
  void beginTargetRegion();

  /// Used for marking end of a target region
  void endTargetRegion();

  // ToDo: mhalk
  void ompt_state_set_helper(void *enter_frame, void *codeptr_ra, int flags,
                             int state);

  // Called by all trace generation routines
  void set_trace_record_common(ompt_record_ompt_t *data_ptr,
                               ompt_callbacks_t cbt);
  // Type specific helpers
  void set_trace_record_target_data_op(ompt_record_target_data_op_t *rec,
                                       ompt_target_data_op_t data_op,
                                       void *src_addr, int64_t src_device_num,
                                       void *dest_ptr, int64_t dest_device_num,
                                       size_t bytes);
  void set_trace_record_target_kernel(ompt_record_target_kernel_t *rec,
                                      unsigned int num_teams);
  void set_trace_record_target(ompt_record_target_t *rec, int64_t device_id,
                               ompt_target_t kind,
                               ompt_scope_endpoint_t endpoint, void *code);
};

/// Thread local state for target region and associated metadata
extern thread_local llvm::omp::target::ompt::Interface OmptInterface;

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm
#else
#define OMPT_IF_BUILT(stmt)
#endif

#endif // _OMPTARGET_OMPTINTERFACE_H
