//===---- OmptTracing.h - Target independent OMPT callbacks --*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface used by target-independent runtimes to coordinate registration and
// invocation of OMPT tracing functionality.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_INCLUDE_OMPTTRACING_H
#define OPENMP_LIBOMPTARGET_INCLUDE_OMPTTRACING_H

#ifdef OMPT_SUPPORT

#include <unordered_map>

#include "OmptCommonDefs.h"
#include "OmptTracingBuffer.h"

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

namespace llvm {
namespace omp {
namespace target {
namespace ompt {

/// A tool may register unique buffer-request and buffer-completion
/// callback functions for a device. The following are utility functions to
/// manage those functions.

/// Given a device-id, return the corresponding buffer-request callback
/// function.
ompt_callback_buffer_request_t getBufferRequestFn(int DeviceId);

/// Give a device-id, return the corresponding buffer-completion callback
/// function.
ompt_callback_buffer_complete_t getBufferCompleteFn(int DeviceId);

/// Given a device-id, set the corresponding buffer-request and
/// buffer-completion callback functions.
void setBufferManagementFns(int DeviceId, ompt_callback_buffer_request_t ReqFn,
                            ompt_callback_buffer_complete_t CmpltFn);

/// Given a device-id, remove the corresponding buffer-request and
/// buffer-completion callback functions.
void removeBufferManagementFns(int DeviceId);

/// Is device tracing stopped for all devices?
bool isAllDeviceTracingStopped();

/// Invoke callback function for buffer request events
void ompt_callback_buffer_request(int DeviceId, ompt_buffer_t **BufferPtr,
                                  size_t *Bytes);

/// Invoke callback function for buffer complete events
void ompt_callback_buffer_complete(int DeviceId, ompt_buffer_t *Buffer,
                                   size_t Bytes,
                                   ompt_buffer_cursor_t BeginCursor,
                                   int BufferOwned);

/// Set 'start' and 'stop' for the current trace record
void setOmptTimestamp(uint64_t StartTime, uint64_t EndTime);

/// Set the linear function correlation between host and device clocks
void setOmptHostToDeviceRate(double Slope, double Offset);

/// Set / store the number of granted teams
void setOmptGrantedNumTeams(uint64_t NumTeams);

/// Activate / deactivate tracing
void setTracingState(bool Enabled);

/// Check if the given tracing type is monitored
bool isTracingTypeEnabled(unsigned int EventTy);

/// Set whether the given tracing type should be monitored (or not)
void setTracingTypeEnabled(unsigned int EventTy, bool Enable);

/// Set / reset the given tracing types (EventTy = 0 corresponds to 'all')
ompt_set_result_t setTraceEventTy(ompt_device_t *Device, unsigned int Enable,
                                  unsigned int EventTy);

/// Return thread id
uint64_t getThreadId();

// Mutexes to serialize invocation of device-independent entry points
extern std::mutex TraceAccessMutex;
extern std::mutex TraceControlMutex;

// Ensure serialization of calls to std::hash
extern std::mutex TraceHashThreadMutex;

// Protect map from device-id to the corresponding buffer-request and
// buffer-completion callback functions.
extern std::mutex BufferManagementFnMutex;

// Map from device-id to the corresponding buffer-request and buffer-completion
// callback functions.
extern std::unordered_map<int, std::pair<ompt_callback_buffer_request_t,
                                         ompt_callback_buffer_complete_t>>
    BufferManagementFns;

// Thread local variables used by the plugin to communicate OMPT information
// that are then used to populate trace records. This method assumes a
// synchronous implementation, otherwise it won't work.
extern thread_local uint32_t TraceRecordNumGrantedTeams;
extern thread_local uint64_t TraceRecordStartTime;
extern thread_local uint64_t TraceRecordStopTime;

// Thread local thread-id.
extern thread_local uint64_t ThreadId;

// Manage all tracing records in one place
extern OmptTracingBufferMgr TraceRecordManager;

// Keep track of enabled tracing event types
extern std::atomic<uint64_t> TracingTypesEnabled;

/// OMPT tracing status; (Re-)Set via 'setTracingState'
extern bool TracingActive;

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm

#pragma pop_macro("DEBUG_PREFIX")

#endif // OMPT_SUPPORT

#endif // OPENMP_LIBOMPTARGET_INCLUDE_OMPTTRACING_H
