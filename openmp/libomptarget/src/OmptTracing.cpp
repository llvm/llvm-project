//===-- OmptTracing.cpp - Target independent OpenMP target RTL --- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of OMPT tracing interfaces for target independent layer
//
//===----------------------------------------------------------------------===//

#ifdef OMPT_SUPPORT

#include "OmptTracing.h"
#include "Shared/Debug.h"
#include "OpenMP/OMPT/Callback.h"
#include "OpenMP/OMPT/Interface.h"
#include "OmptTracingBuffer.h"
#include "omp-tools.h"
#include "private.h"

#include "llvm/Support/DynamicLibrary.h"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

#define isTracingTypeDisabled(TracingType)                                     \
  (!llvm::omp::target::ompt::TracingActive ||                                  \
   (!llvm::omp::target::ompt::isTracingTypeEnabled(TracingType) &&             \
    !llvm::omp::target::ompt::isTracingTypeEnabled(TracingType##_emi)))

using namespace llvm::omp::target::ompt;

OmptTracingBufferMgr llvm::omp::target::ompt::TraceRecordManager;

std::mutex llvm::omp::target::ompt::TraceAccessMutex;
std::mutex llvm::omp::target::ompt::TraceControlMutex;
std::mutex llvm::omp::target::ompt::TraceHashThreadMutex;
std::mutex llvm::omp::target::ompt::BufferManagementFnMutex;

std::unordered_map<int /*DeviceId*/, std::pair<ompt_callback_buffer_request_t,
                                               ompt_callback_buffer_complete_t>>
    llvm::omp::target::ompt::BufferManagementFns;

thread_local uint32_t llvm::omp::target::ompt::TraceRecordNumGrantedTeams = 0;
thread_local uint64_t llvm::omp::target::ompt::TraceRecordStartTime = 0;
thread_local uint64_t llvm::omp::target::ompt::TraceRecordStopTime = 0;
thread_local uint64_t llvm::omp::target::ompt::ThreadId =
    std::numeric_limits<uint64_t>::max();

std::atomic<uint64_t> llvm::omp::target::ompt::TracingTypesEnabled{0};

bool llvm::omp::target::ompt::TracingActive = false;

ompt_callback_buffer_request_t
llvm::omp::target::ompt::getBufferRequestFn(int DeviceId) {
  std::unique_lock<std::mutex> Lock(BufferManagementFnMutex);
  auto BufferMgrItr = BufferManagementFns.find(DeviceId);
  if (BufferMgrItr == BufferManagementFns.end()) {
    return nullptr;
  }
  return BufferMgrItr->second.first;
}

ompt_callback_buffer_complete_t
llvm::omp::target::ompt::getBufferCompleteFn(int DeviceId) {
  std::unique_lock<std::mutex> Lock(BufferManagementFnMutex);
  auto BufferMgrItr = BufferManagementFns.find(DeviceId);
  if (BufferMgrItr == BufferManagementFns.end()) {
    return nullptr;
  }
  return BufferMgrItr->second.second;
}

void llvm::omp::target::ompt::setBufferManagementFns(
    int DeviceId, ompt_callback_buffer_request_t ReqFn,
    ompt_callback_buffer_complete_t CmpltFn) {
  std::unique_lock<std::mutex> Lock(BufferManagementFnMutex);
  auto BufferMgrItr = BufferManagementFns.find(DeviceId);
  if (BufferMgrItr != BufferManagementFns.end()) {
    REPORT("Buffer request and complete functions already exist for device %d, "
           "ignoring ...\n",
           DeviceId);
    return;
  }
  BufferManagementFns[DeviceId] = std::make_pair(ReqFn, CmpltFn);
}

void llvm::omp::target::ompt::removeBufferManagementFns(int DeviceId) {
  std::unique_lock<std::mutex> Lock(BufferManagementFnMutex);
  auto BufferMgrItr = BufferManagementFns.find(DeviceId);
  if (BufferMgrItr == BufferManagementFns.end()) {
    REPORT("Buffer request and complete functions don't exist for device %d, "
           "ignoring ...\n",
           DeviceId);
    return;
  }
  BufferManagementFns.erase(BufferMgrItr);
}

bool llvm::omp::target::ompt::isAllDeviceTracingStopped() {
  std::unique_lock<std::mutex> Lock(BufferManagementFnMutex);
  return BufferManagementFns.empty();
}

void llvm::omp::target::ompt::ompt_callback_buffer_request(
    int DeviceId, ompt_buffer_t **BufferPtr, size_t *Bytes) {
  if (auto Fn = getBufferRequestFn(DeviceId))
    Fn(DeviceId, BufferPtr, Bytes);
}

void llvm::omp::target::ompt::ompt_callback_buffer_complete(
    int DeviceId, ompt_buffer_t *Buffer, size_t Bytes,
    ompt_buffer_cursor_t BeginCursor, int BufferOwned) {
  if (auto Fn = getBufferCompleteFn(DeviceId))
    Fn(DeviceId, Buffer, Bytes, BeginCursor, BufferOwned);
}

void llvm::omp::target::ompt::setTracingState(bool State) {
  TracingActive = State;
}

bool llvm::omp::target::ompt::isTracingTypeEnabled(unsigned int EventTy) {
  // Make sure we do not shift more than std::numeric_limits<uint64_t>::digits
  assert(EventTy < 64);
  if (EventTy < 64)
    return (TracingTypesEnabled & (1UL << EventTy)) != 0;
  return false;
}

void llvm::omp::target::ompt::setTracingTypeEnabled(unsigned int EventTy,
                                                    bool Enable) {
  // Make sure we do not shift more than std::numeric_limits<uint64_t>::digits
  assert(EventTy < 64);
  if (EventTy < 64) {
    if (Enable)
      TracingTypesEnabled |= (1UL << EventTy);
    else
      TracingTypesEnabled &= ~(1UL << EventTy);
  }
}

ompt_set_result_t llvm::omp::target::ompt::setTraceEventTy(
    ompt_device_t *Device, unsigned int Enable, unsigned int EventTy) {
  // TODO handle device

  DP("Executing setTraceEventTy: Device=%p Enable=%d EventTy=%d\n", Device,
     Enable, EventTy);

  bool isEventTyEnabled = Enable > 0;
  if (EventTy == 0) {
    // Set / reset all supported types
    setTracingTypeEnabled(ompt_callbacks_t::ompt_callback_target,
                          isEventTyEnabled);
    setTracingTypeEnabled(ompt_callbacks_t::ompt_callback_target_data_op,
                          isEventTyEnabled);
    setTracingTypeEnabled(ompt_callbacks_t::ompt_callback_target_submit,
                          isEventTyEnabled);
    setTracingTypeEnabled(ompt_callbacks_t::ompt_callback_target_emi,
                          isEventTyEnabled);
    setTracingTypeEnabled(ompt_callbacks_t::ompt_callback_target_data_op_emi,
                          isEventTyEnabled);
    setTracingTypeEnabled(ompt_callbacks_t::ompt_callback_target_submit_emi,
                          isEventTyEnabled);

    if (isEventTyEnabled) {
      // Event subset is enabled
      return ompt_set_sometimes;
    } else {
      // All events are disabled
      return ompt_set_always;
    }
  }

  switch (EventTy) {
  case ompt_callbacks_t::ompt_callback_target:
  case ompt_callbacks_t::ompt_callback_target_data_op:
  case ompt_callbacks_t::ompt_callback_target_submit:
  case ompt_callbacks_t::ompt_callback_target_emi:
  case ompt_callbacks_t::ompt_callback_target_data_op_emi:
  case ompt_callbacks_t::ompt_callback_target_submit_emi: {
    setTracingTypeEnabled(EventTy, isEventTyEnabled);
    return ompt_set_always;
  }
  default: {
    if (isEventTyEnabled) {
      // Unimplemented
      return ompt_set_never;
    } else {
      // Always disabled anyways
      return ompt_set_always;
    }
  }
  }
}

uint64_t llvm::omp::target::ompt::getThreadId() {
  // Grab the value from thread local storage, if valid.
  if (ThreadId != std::numeric_limits<uint64_t>::max())
    return ThreadId;
  // Otherwise set it, protecting the hash with a lock.
  std::unique_lock<std::mutex> Lock(TraceHashThreadMutex);
  ThreadId = std::hash<std::thread::id>()(std::this_thread::get_id());
  return ThreadId;
}

void Interface::setTraceRecordCommon(ompt_record_ompt_t *DataPtr,
                                     ompt_callbacks_t CallbackType) {
  DataPtr->type = CallbackType;

  if (CallbackType == ompt_callback_target)
    DataPtr->time = 0; // Currently, no consumer, so no need to set it
  else
    DataPtr->time = TraceRecordStartTime;

  DataPtr->thread_id = getThreadId();
  DataPtr->target_id = TargetData.value;
}

void Interface::setTraceRecordTargetDataOp(ompt_record_target_data_op_t *Record,
                                           ompt_target_data_op_t DataOpType,
                                           void *SrcAddr, int64_t SrcDeviceNum,
                                           void *DstAddr, int64_t DstDeviceNum,
                                           size_t Bytes, void *CodePtr) {
  Record->host_op_id = HostOpId;
  Record->optype = DataOpType;
  Record->src_addr = SrcAddr;
  Record->src_device_num = SrcDeviceNum;
  Record->dest_addr = DstAddr;
  Record->dest_device_num = DstDeviceNum;
  Record->bytes = Bytes;
  Record->end_time = TraceRecordStopTime;
  Record->codeptr_ra = CodePtr;
}

void Interface::setTraceRecordTargetKernel(ompt_record_target_kernel_t *Record,
                                           unsigned int NumTeams) {
  Record->host_op_id = HostOpId;
  Record->requested_num_teams = NumTeams;
  Record->granted_num_teams = TraceRecordNumGrantedTeams;
  Record->end_time = TraceRecordStopTime;
}

void Interface::setTraceRecordTarget(ompt_record_target_t *Record,
                                     int64_t DeviceId, ompt_target_t TargetKind,
                                     ompt_scope_endpoint_t Endpoint,
                                     void *CodePtr) {
  Record->kind = TargetKind;
  Record->endpoint = Endpoint;
  Record->device_num = DeviceId;
  assert(TaskData);
  Record->task_id = TaskData->value;
  Record->target_id = TargetData.value;
  Record->codeptr_ra = CodePtr;
}

void Interface::startTargetDataAllocTrace(int64_t DeviceId, void *HstPtrBegin,
                                          void **TgtPtrBegin, size_t Size,
                                          void *Code) {}

ompt_record_ompt_t *Interface::stopTargetDataAllocTrace(int64_t DeviceId,
                                                        void *HstPtrBegin,
                                                        void **TgtPtrBegin,
                                                        size_t Size,
                                                        void *Code) {
  if (isTracingTypeDisabled(ompt_callback_target_data_op))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target_data_op, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target_data_op);
  setTraceRecordTargetDataOp(&DataPtr->record.target_data_op,
                             ompt_target_data_alloc, HstPtrBegin,
                             /*SrcDeviceNum=*/omp_get_initial_device(),
                             *TgtPtrBegin, DeviceId, Size, Code);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target_data_submit trace record %p\n", DataPtr);
  return DataPtr;
}

void Interface::startTargetDataDeleteTrace(int64_t DeviceId, void *TgtPtrBegin,
                                           void *Code) {}

ompt_record_ompt_t *Interface::stopTargetDataDeleteTrace(int64_t DeviceId,
                                                         void *TgtPtrBegin,
                                                         void *Code) {
  if (isTracingTypeDisabled(ompt_callback_target_data_op))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target_data_op, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target_data_op);
  setTraceRecordTargetDataOp(&DataPtr->record.target_data_op,
                             ompt_target_data_delete, TgtPtrBegin, DeviceId,
                             /*DstAddr=*/nullptr,
                             /*DstDeviceNum=*/-1, /*Bytes=*/0, Code);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target_data_submit trace record %p\n", DataPtr);
  return DataPtr;
}

void Interface::startTargetDataSubmitTrace(int64_t SrcDeviceId,
                                           void *SrcPtrBegin,
                                           int64_t DstDeviceId,
                                           void *DstPtrBegin, size_t Size,
                                           void *Code) {}

ompt_record_ompt_t *
Interface::stopTargetDataSubmitTrace(int64_t SrcDeviceId, void *SrcPtrBegin,
                                     int64_t DstDeviceId, void *DstPtrBegin,
                                     size_t Size, void *Code) {
  if (isTracingTypeDisabled(ompt_callback_target_data_op))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target_data_op, DstDeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target_data_op);
  setTraceRecordTargetDataOp(&DataPtr->record.target_data_op,
                             ompt_target_data_transfer_to_device, SrcPtrBegin,
                             SrcDeviceId, DstPtrBegin, DstDeviceId, Size, Code);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target_data_submit trace record %p\n", DataPtr);
  return DataPtr;
}

void Interface::startTargetDataRetrieveTrace(int64_t SrcDeviceId,
                                             void *SrcPtrBegin,
                                             int64_t DstDeviceId,
                                             void *DstPtrBegin, size_t Size,
                                             void *Code) {}

ompt_record_ompt_t *
Interface::stopTargetDataRetrieveTrace(int64_t SrcDeviceId, void *SrcPtrBegin,
                                       int64_t DstDeviceId, void *DstPtrBegin,
                                       size_t Size, void *Code) {
  if (isTracingTypeDisabled(ompt_callback_target_data_op))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target_data_op, SrcDeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target_data_op);
  setTraceRecordTargetDataOp(&DataPtr->record.target_data_op,
                             ompt_target_data_transfer_from_device, SrcPtrBegin,
                             SrcDeviceId, DstPtrBegin, DstDeviceId, Size, Code);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target_data_submit trace record %p\n", DataPtr);
  return DataPtr;
}

void Interface::startTargetSubmitTrace(int64_t DeviceId,
                                       unsigned int NumTeams) {}

ompt_record_ompt_t *Interface::stopTargetSubmitTrace(int64_t DeviceId,
                                                     unsigned int NumTeams) {
  if (isTracingTypeDisabled(ompt_callback_target_submit))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target_submit, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target_submit);
  setTraceRecordTargetKernel(&DataPtr->record.target_kernel, NumTeams);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target_submit trace record %p\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::startTargetDataEnterTrace(int64_t DeviceId,
                                                         void *CodePtr) {
  if (isTracingTypeDisabled(ompt_callback_target))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target);
  setTraceRecordTarget(&DataPtr->record.target, DeviceId,
                       ompt_target_enter_data, ompt_scope_begin, CodePtr);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target trace record %p, completing a kernel\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::stopTargetDataEnterTrace(int64_t DeviceId,
                                                        void *CodePtr) {
  if (isTracingTypeDisabled(ompt_callback_target))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target);
  setTraceRecordTarget(&DataPtr->record.target, DeviceId,
                       ompt_target_enter_data, ompt_scope_end, CodePtr);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target trace record %p, completing a kernel\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::startTargetDataExitTrace(int64_t DeviceId,
                                                        void *CodePtr) {
  if (isTracingTypeDisabled(ompt_callback_target))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target);
  setTraceRecordTarget(&DataPtr->record.target, DeviceId, ompt_target_exit_data,
                       ompt_scope_begin, CodePtr);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target trace record %p, completing a kernel\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::stopTargetDataExitTrace(int64_t DeviceId,
                                                       void *CodePtr) {
  if (isTracingTypeDisabled(ompt_callback_target))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target);
  setTraceRecordTarget(&DataPtr->record.target, DeviceId, ompt_target_exit_data,
                       ompt_scope_end, CodePtr);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target trace record %p, completing a kernel\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::startTargetUpdateTrace(int64_t DeviceId,
                                                      void *CodePtr) {
  if (isTracingTypeDisabled(ompt_callback_target))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target);
  setTraceRecordTarget(&DataPtr->record.target, DeviceId, ompt_target_update,
                       ompt_scope_begin, CodePtr);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target trace record %p, completing a kernel\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::stopTargetUpdateTrace(int64_t DeviceId,
                                                     void *CodePtr) {
  if (isTracingTypeDisabled(ompt_callback_target))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target);
  setTraceRecordTarget(&DataPtr->record.target, DeviceId, ompt_target_update,
                       ompt_scope_end, CodePtr);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target trace record %p, completing a kernel\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::startTargetTrace(int64_t DeviceId,
                                                void *CodePtr) {
  if (isTracingTypeDisabled(ompt_callback_target))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target);
  setTraceRecordTarget(&DataPtr->record.target, DeviceId, ompt_target,
                       ompt_scope_begin, CodePtr);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("Generated target trace record %p, completing a kernel\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::stopTargetTrace(int64_t DeviceId,
                                               void *CodePtr) {
  if (isTracingTypeDisabled(ompt_callback_target))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target, DeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target);
  setTraceRecordTarget(&DataPtr->record.target, DeviceId, ompt_target,
                       ompt_scope_end, CodePtr);

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);

  DP("Generated target trace record %p, completing a kernel\n", DataPtr);

  return DataPtr;
}

extern "C" {
// Device-independent entry point for ompt_set_trace_ompt
ompt_set_result_t libomptarget_ompt_set_trace_ompt(ompt_device_t *Device,
                                                   unsigned int Enable,
                                                   unsigned int EventTy) {
  std::unique_lock<std::mutex> Lock(TraceAccessMutex);
  return llvm::omp::target::ompt::setTraceEventTy(Device, Enable, EventTy);
}

// Device-independent entry point for ompt_start_trace
int libomptarget_ompt_start_trace(int DeviceId,
                                  ompt_callback_buffer_request_t Request,
                                  ompt_callback_buffer_complete_t Complete) {
  std::unique_lock<std::mutex> Lock(TraceControlMutex);
  if (Request && Complete) {
    // Set buffer related functions
    llvm::omp::target::ompt::setBufferManagementFns(DeviceId, Request,
                                                    Complete);
    llvm::omp::target::ompt::setTracingState(/*Enabled=*/true);
    TraceRecordManager.startHelperThreads();
    // Success
    return 1;
  }
  // Failure
  return 0;
}

// Device-independent entry point for ompt_flush_trace
int libomptarget_ompt_flush_trace(int DeviceId) {
  std::unique_lock<std::mutex> Lock(TraceControlMutex);
  return TraceRecordManager.flushAllBuffers(DeviceId);
}

// Device independent entry point for ompt_stop_trace
int libomptarget_ompt_stop_trace(int DeviceId) {
  std::unique_lock<std::mutex> Lock(TraceControlMutex);

  // Schedule flushing of trace records for this device
  int Status = TraceRecordManager.flushAllBuffers(DeviceId);

  // De-register this device so that no more traces are collected
  // or delivered for this device until an ompt_start_trace is
  // invoked for this device.
  removeBufferManagementFns(DeviceId);

  // If no device is being traced, shut down the helper threads. A
  // subsequent ompt_start_trace will start up the helper threads.
  if (isAllDeviceTracingStopped()) {
    // TODO shutdown should perhaps return a status
    TraceRecordManager.shutdownHelperThreads();
    llvm::omp::target::ompt::setTracingState(/*Enabled=*/false);
  }
  return Status;
}

// Device independent entry point for ompt_advance_buffer_cursor
// Note: The input parameter size is unused here. It refers to the
// bytes returned in the corresponding callback.
int libomptarget_ompt_advance_buffer_cursor(ompt_device_t *Device,
                                            ompt_buffer_t *Buffer, size_t Size,
                                            ompt_buffer_cursor_t CurrentPos,
                                            ompt_buffer_cursor_t *NextPos) {
  char *TraceRecord = (char *)CurrentPos;
  // Don't assert if CurrentPos is null, just indicate end of buffer
  if (TraceRecord == nullptr || TraceRecordManager.isLastCursor(TraceRecord)) {
    *NextPos = 0;
    return false;
  }
  // TODO In debug mode, assert that the metadata points to the
  // input parameter buffer

  size_t TRSize = TraceRecordManager.getTRSize();
  *NextPos = (ompt_buffer_cursor_t)(TraceRecord + TRSize);
  DP("Advanced buffer pointer by %lu bytes to %p\n", TRSize,
     TraceRecord + TRSize);
  return true;
}

// This function is invoked before the kernel launch. So, when the trace record
// is populated after kernel completion, TraceRecordNumGrantedTeams is already
// updated.
void libomptarget_ompt_set_granted_teams(uint32_t NumTeams) {
  TraceRecordNumGrantedTeams = NumTeams;
}

// Assume a synchronous implementation and set thread local variables to track
// timestamps. The thread local variables can then be used to populate trace
// records.
void libomptarget_ompt_set_timestamp(uint64_t Start, uint64_t Stop) {
  TraceRecordStartTime = Start;
  TraceRecordStopTime = Stop;
}

// Device-independent entry point to query for the trace format used.
// Currently, only OMPT format is supported.
ompt_record_t
libomptarget_ompt_get_record_type(ompt_buffer_t *Buffer,
                                  ompt_buffer_cursor_t CurrentPos) {
  // TODO: When different OMPT trace buffer formats supported, this needs to be
  // fixed.
  return ompt_record_t::ompt_record_ompt;
}
} // extern "C"

#pragma pop_macro("DEBUG_PREFIX")

#endif // OMPT_SUPPORT
