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

using namespace llvm::omp::target::ompt;

OmptTracingBufferMgr llvm::omp::target::ompt::TraceRecordManager;

std::mutex llvm::omp::target::ompt::DeviceAccessMutex;
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

std::map<int32_t, uint64_t> llvm::omp::target::ompt::TracedDevices;

bool llvm::omp::target::ompt::TracingActive = false;

void llvm::omp::target::ompt::resetTimestamp(uint64_t *T) { *T = 0; }

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

inline void setDeviceTracing(uint64_t &TracingTypes) {
  // Set bit 0 to indicate generally enabled device tracing.
  TracingTypes |= 1UL;
}

inline void resetDeviceTracing(uint64_t &TracingTypes) {
  // Reset bit 0 to indicate generally disabled device tracing.
  TracingTypes &= ~(1UL);
}

inline bool checkDeviceTracingState(const uint64_t &TracingTypes) {
  // Return state of bit 0 to indicate if device is actively traced.
  return TracingTypes & 1UL;
}

void llvm::omp::target::ompt::enableDeviceTracing(int DeviceId) {
  std::unique_lock<std::mutex> Lock(DeviceAccessMutex);
  auto Device = TracedDevices.find(DeviceId);
  if (Device == TracedDevices.end()) {
    uint64_t TracingTypes{0};
    setDeviceTracing(TracingTypes);
    TracedDevices.emplace(DeviceId, TracingTypes);
  } else
    setDeviceTracing(Device->second);
  // In any case: at least one device is traced
  TracingActive = true;
}

void llvm::omp::target::ompt::disableDeviceTracing(int DeviceId) {
  std::unique_lock<std::mutex> Lock(DeviceAccessMutex);
  auto Device = TracedDevices.find(DeviceId);
  if (Device == TracedDevices.end()) {
    uint64_t TracingTypes{0};
    resetDeviceTracing(TracingTypes);
    TracedDevices.emplace(DeviceId, TracingTypes);
  } else
    resetDeviceTracing(Device->second);

  // Check for actively traced devices
  for (auto &Dev : TracedDevices)
    if (checkDeviceTracingState(Dev.second))
      return;

  // If no device is currently traced: set global tracing flag to false
  TracingActive = false;
}

bool llvm::omp::target::ompt::isTracingEnabled(int DeviceId,
                                               unsigned int EventTy) {
  return TracingActive && isTracedDevice(DeviceId) &&
         isTracingTypeGroupEnabled(DeviceId, EventTy);
}

bool llvm::omp::target::ompt::isTracedDevice(int DeviceId) {
  std::unique_lock<std::mutex> Lock(DeviceAccessMutex);
  auto Device = TracedDevices.find(DeviceId);
  if (Device != TracedDevices.end())
    return checkDeviceTracingState(Device->second);

  return false;
}

bool llvm::omp::target::ompt::isTracingTypeEnabled(int DeviceId,
                                                   unsigned int EventTy) {
  std::unique_lock<std::mutex> Lock(DeviceAccessMutex);
  // Make sure we do not shift more than std::numeric_limits<uint64_t>::digits
  assert(EventTy < 64 && "Shift limit exceeded: EventTy must be less than 64");
  auto Device = TracedDevices.find(DeviceId);
  if (Device != TracedDevices.end() && EventTy < 64)
    return (Device->second & (1UL << EventTy));
  return false;
}

bool llvm::omp::target::ompt::isTracingTypeGroupEnabled(int DeviceId,
                                                        unsigned int EventTy) {
  std::unique_lock<std::mutex> Lock(DeviceAccessMutex);
  // Make sure we do not shift more than std::numeric_limits<uint64_t>::digits
  assert(EventTy < 64 && "Shift limit exceeded: EventTy must be less than 64");
  auto Device = TracedDevices.find(DeviceId);
  if (Device != TracedDevices.end() && EventTy < 64) {
    auto TracedEvents = Device->second;
    switch (EventTy) {
    case ompt_callbacks_t::ompt_callback_target:
    case ompt_callbacks_t::ompt_callback_target_emi:
      return ((TracedEvents & (1UL << ompt_callback_target))) ||
             ((TracedEvents & (1UL << ompt_callback_target_emi)));
    case ompt_callbacks_t::ompt_callback_target_data_op:
    case ompt_callbacks_t::ompt_callback_target_data_op_emi:
      return ((TracedEvents & (1UL << ompt_callback_target_data_op))) ||
             ((TracedEvents & (1UL << ompt_callback_target_data_op_emi)));
    case ompt_callbacks_t::ompt_callback_target_submit:
    case ompt_callbacks_t::ompt_callback_target_submit_emi:
      return ((TracedEvents & (1UL << ompt_callback_target_submit))) ||
             ((TracedEvents & (1UL << ompt_callback_target_submit_emi)));
    // Special case: EventTy == 0 -> Check all EventTy
    case 0:
      return ((TracedEvents & (1UL << ompt_callback_target))) ||
             ((TracedEvents & (1UL << ompt_callback_target_emi))) ||
             ((TracedEvents & (1UL << ompt_callback_target_data_op))) ||
             ((TracedEvents & (1UL << ompt_callback_target_data_op_emi))) ||
             ((TracedEvents & (1UL << ompt_callback_target_submit))) ||
             ((TracedEvents & (1UL << ompt_callback_target_submit_emi)));
    }
  }
  return false;
}

void llvm::omp::target::ompt::setTracingTypeEnabled(uint64_t &TracedEventTy,
                                                    bool Enable,
                                                    unsigned int EventTy) {
  // Make sure we do not shift more than std::numeric_limits<uint64_t>::digits
  assert(EventTy < 64 && "Shift limit exceeded: EventTy must be less than 64");
  if (EventTy < 64) {
    if (Enable)
      TracedEventTy |= (1UL << EventTy);
    else
      TracedEventTy &= ~(1UL << EventTy);
  }
}

ompt_set_result_t
llvm::omp::target::ompt::setTraceEventTy(int DeviceId, unsigned int Enable,
                                         unsigned int EventTy) {
  if (DeviceId < 0) {
    REPORT("Failed to set trace event type for DeviceId=%d\n", DeviceId);
    return ompt_set_never;
  }

  DP("Executing setTraceEventTy: DeviceId=%d Enable=%d EventTy=%d\n", DeviceId,
     Enable, EventTy);

  std::unique_lock<std::mutex> Lock(DeviceAccessMutex);
  if (TracedDevices.find(DeviceId) == TracedDevices.end())
    TracedDevices.emplace(DeviceId, 0UL);

  auto &TracedEventTy = TracedDevices[DeviceId];
  bool Enabled = Enable > 0;
  if (EventTy == 0) {
    // Set / reset all supported types
    setTracingTypeEnabled(TracedEventTy, Enabled,
                          ompt_callbacks_t::ompt_callback_target);
    setTracingTypeEnabled(TracedEventTy, Enabled,
                          ompt_callbacks_t::ompt_callback_target_data_op);
    setTracingTypeEnabled(TracedEventTy, Enabled,
                          ompt_callbacks_t::ompt_callback_target_submit);
    setTracingTypeEnabled(TracedEventTy, Enabled,
                          ompt_callbacks_t::ompt_callback_target_emi);
    setTracingTypeEnabled(TracedEventTy, Enabled,
                          ompt_callbacks_t::ompt_callback_target_data_op_emi);
    setTracingTypeEnabled(TracedEventTy, Enabled,
                          ompt_callbacks_t::ompt_callback_target_submit_emi);

    if (Enabled) {
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
    setTracingTypeEnabled(TracedEventTy, Enabled, EventTy);
    return ompt_set_always;
  }
  default: {
    if (Enabled) {
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
  else {
    DataPtr->time = TraceRecordStartTime;
    resetTimestamp(&TraceRecordStartTime);
  }

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
  resetTimestamp(&TraceRecordStopTime);

  Record->codeptr_ra = CodePtr;
}

void Interface::setTraceRecordTargetKernel(ompt_record_target_kernel_t *Record,
                                           unsigned int NumTeams) {
  Record->host_op_id = HostOpId;
  Record->requested_num_teams = NumTeams;
  Record->granted_num_teams = TraceRecordNumGrantedTeams;

  Record->end_time = TraceRecordStopTime;
  resetTimestamp(&TraceRecordStopTime);
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
  if (!isTracingEnabled(DeviceId, ompt_callback_target_data_op))
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
  DP("Generated trace record: %p (ompt_target_data_alloc)\n", DataPtr);
  return DataPtr;
}

void Interface::startTargetDataDeleteTrace(int64_t DeviceId, void *TgtPtrBegin,
                                           void *Code) {}

ompt_record_ompt_t *Interface::stopTargetDataDeleteTrace(int64_t DeviceId,
                                                         void *TgtPtrBegin,
                                                         void *Code) {
  if (!isTracingEnabled(DeviceId, ompt_callback_target_data_op))
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
  DP("Generated trace record: %p (ompt_target_data_delete)\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *
Interface::startTargetDataSubmitTrace(int64_t SrcDeviceId, void *SrcPtrBegin,
                                      int64_t DstDeviceId, void *DstPtrBegin,
                                      size_t Size, void *Code) {
  if (!isTracingEnabled(DstDeviceId, ompt_callback_target_data_op))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target_data_op, DstDeviceId);

  // This event will not be traced
  if (DataPtr == nullptr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target_data_op);
  DataPtr->time = 0; // Set to sanity value and let "stop" function fix it

  // Set some of the data-op specific fields here
  setTraceRecordTargetDataOp(&DataPtr->record.target_data_op,
                             ompt_target_data_transfer_to_device, SrcPtrBegin,
                             SrcDeviceId, DstPtrBegin, DstDeviceId, Size, Code);

  DP("OMPT-Async: Returning data trace record buf ptr %p\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *
Interface::startTargetDataRetrieveTrace(int64_t SrcDeviceId, void *SrcPtrBegin,
                                        int64_t DstDeviceId, void *DstPtrBegin,
                                        size_t Size, void *Code) {
  if (!isTracingEnabled(SrcDeviceId, ompt_callback_target_data_op))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target_data_op, SrcDeviceId);

  if (!DataPtr)
    return nullptr;

  setTraceRecordCommon(DataPtr, ompt_callback_target_data_op);
  DataPtr->time = 0; // Set to sanity value and let "stop" function fix it

  // Set some of the data-op specific fields here
  setTraceRecordTargetDataOp(&DataPtr->record.target_data_op,
                             ompt_target_data_transfer_from_device, SrcPtrBegin,
                             SrcDeviceId, DstPtrBegin, DstDeviceId, Size, Code);

  DP("OMPT-Async: Returning data trace record buf ptr %p\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::stopTargetDataMovementTraceAsync(
    ompt_record_ompt_t *DataPtr, uint64_t NanosStart, uint64_t NanosEnd) {
  // Finalize the data that comes from the plugin.
  DataPtr->time = NanosStart;
  auto Record = static_cast<ompt_record_target_data_op_t *>(
      &DataPtr->record.target_data_op);
  Record->end_time = NanosEnd;

  // The trace record has been created, mark it ready for delivery to the tool
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("OMPT-Async: Completed target_data trace record %p\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::startTargetSubmitTrace(int64_t DeviceId,
                                                      unsigned int NumTeams) {
  if (!isTracingEnabled(DeviceId, ompt_callback_target_submit))
    return nullptr;

  ompt_record_ompt_t *DataPtr =
      (ompt_record_ompt_t *)TraceRecordManager.assignCursor(
          ompt_callback_target_submit, DeviceId);

  // Set all known entries and leave remaining to the stop function
  setTraceRecordCommon(DataPtr, ompt_callback_target_submit);
  DataPtr->time = 0; // Set to sanity value and let "stop" function fix it
  // Kernel specific things
  DataPtr->record.target_kernel.requested_num_teams = NumTeams;
  DataPtr->record.target_kernel.host_op_id = getHostOpId();

  // May be null if event is not traced
  DP("OMPT-Async: Returning kernel trace record buf ptr %p\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *
Interface::stopTargetSubmitTraceAsync(ompt_record_ompt_t *DataPtr,
                                      unsigned int NumTeams,
                                      uint64_t NanosStart, uint64_t NanosStop) {
  // Common fields
  DataPtr->time = NanosStart;
  // Submit specific
  DataPtr->record.target_kernel.end_time = NanosStop;
  DataPtr->record.target_kernel.granted_num_teams = NumTeams;

  // Ready Record
  TraceRecordManager.setTRStatus(DataPtr, OmptTracingBufferMgr::TR_ready);
  DP("OMPT-Async: Completed trace record buf ptr %p\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::startTargetDataEnterTrace(int64_t DeviceId,
                                                         void *CodePtr) {
  if (!isTracingEnabled(DeviceId, ompt_callback_target))
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
  DP("Returning trace record buf ptr: %p (ompt_target_enter_data)\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::stopTargetDataEnterTrace(int64_t DeviceId,
                                                        void *CodePtr) {
  if (!isTracingEnabled(DeviceId, ompt_callback_target))
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
  DP("Generated trace record: %p (ompt_target_enter_data)\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::startTargetDataExitTrace(int64_t DeviceId,
                                                        void *CodePtr) {
  if (!isTracingEnabled(DeviceId, ompt_callback_target))
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
  DP("Returning trace record buf ptr: %p (ompt_target_exit_data)\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::stopTargetDataExitTrace(int64_t DeviceId,
                                                       void *CodePtr) {
  if (!isTracingEnabled(DeviceId, ompt_callback_target))
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
  DP("Generated trace record: %p (ompt_target_exit_data)\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::startTargetUpdateTrace(int64_t DeviceId,
                                                      void *CodePtr) {
  if (!isTracingEnabled(DeviceId, ompt_callback_target))
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
  DP("Returning trace record buf ptr: %p (ompt_target_update)\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::stopTargetUpdateTrace(int64_t DeviceId,
                                                     void *CodePtr) {
  if (!isTracingEnabled(DeviceId, ompt_callback_target))
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
  DP("Generated trace record: %p (ompt_target_update)\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::startTargetTrace(int64_t DeviceId,
                                                void *CodePtr) {
  if (!isTracingEnabled(DeviceId, ompt_callback_target))
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
  DP("Returning trace record buf ptr: %p (ompt_target)\n", DataPtr);
  return DataPtr;
}

ompt_record_ompt_t *Interface::stopTargetTrace(int64_t DeviceId,
                                               void *CodePtr) {
  if (!isTracingEnabled(DeviceId, ompt_callback_target))
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

  DP("Generated trace record: %p (ompt_target)\n", DataPtr);
  return DataPtr;
}

extern "C" {
// Device-independent entry point for ompt_set_trace_ompt
ompt_set_result_t libomptarget_ompt_set_trace_ompt(int DeviceId,
                                                   unsigned int Enable,
                                                   unsigned int EventTy) {
  std::unique_lock<std::mutex> Lock(TraceAccessMutex);
  return llvm::omp::target::ompt::setTraceEventTy(DeviceId, Enable, EventTy);
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
    llvm::omp::target::ompt::enableDeviceTracing(DeviceId);
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
    llvm::omp::target::ompt::disableDeviceTracing(DeviceId);
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
