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

#include "omp-tools.h"

// If target OMPT support is compiled in
#ifdef OMPT_SUPPORT
#define OMPT_IF_BUILT(stmt) stmt
#else
#define OMPT_IF_BUILT(stmt)
#endif

/// Callbacks for target regions require task_data representing the
/// encountering task.
/// Callbacks for target regions and target data ops require
/// target_task_data representing the target task region.
typedef ompt_data_t *(*ompt_get_task_data_t)();
typedef ompt_data_t *(*ompt_get_target_task_data_t)();

namespace llvm {
namespace omp {
namespace target {
namespace ompt {

/// Function pointers that will be used to track task_data and
/// target_task_data.
static ompt_get_task_data_t ompt_get_task_data_fn;
static ompt_get_target_task_data_t ompt_get_target_task_data_fn;

/// Used to maintain execution state for this thread
class Interface {
public:
  /// Top-level function for invoking callback before device data allocation
  void beginTargetDataAlloc(int64_t DeviceId, void *TgtPtrBegin, size_t Size,
                            void *Code);

  /// Top-level function for invoking callback after device data allocation
  void endTargetDataAlloc(int64_t DeviceId, void *TgtPtrBegin, size_t Size,
                          void *Code);

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

  /// Used for marking begin of a data operation
  void beginTargetDataOperation();

  /// Used for marking end of a data operation
  void endTargetDataOperation();

  /// Used for marking begin of a target region
  void beginTargetRegion();

  /// Used for marking end of a target region
  void endTargetRegion();
};

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm

extern thread_local llvm::omp::target::ompt::Interface OmptInterface;

#endif // _OMPTARGET_OMPTINTERFACE_H
