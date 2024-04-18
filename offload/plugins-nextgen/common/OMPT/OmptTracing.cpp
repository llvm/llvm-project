//===-- OmptTracing.cpp - Target independent OpenMP target RTL --- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of OMPT tracing interfaces for PluginInterface
//
//===----------------------------------------------------------------------===//

#ifdef OMPT_SUPPORT

#include "Shared/Debug.h"
#include "OmptDeviceTracing.h"
#include "omp-tools.h"

#include "llvm/Support/DynamicLibrary.h"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

// Define OMPT device tracing function entry points
#define defineOmptTracingFn(Name)                                              \
  libomptarget_##Name##_t llvm::omp::target::ompt::Name##_fn = nullptr;
FOREACH_OMPT_DEVICE_TRACING_FN_IMPLEMENTAIONS(defineOmptTracingFn)
#undef defineOmptTracingFn

// Define OMPT device tracing function mutexes
#define defineOmptTracingFnMutex(Name)                                         \
  std::mutex llvm::omp::target::ompt::Name##_mutex;
FOREACH_OMPT_DEVICE_TRACING_FN_IMPLEMENTAIONS(defineOmptTracingFnMutex)
#undef defineOmptTracingFnMutex

std::mutex llvm::omp::target::ompt::DeviceIdWritingMutex;

using namespace llvm::omp::target::ompt;

std::shared_ptr<llvm::sys::DynamicLibrary>
    llvm::omp::target::ompt::ParentLibrary(nullptr);

double llvm::omp::target::ompt::HostToDeviceSlope = .0;
double llvm::omp::target::ompt::HostToDeviceOffset = .0;

std::map<ompt_device_t *, int32_t> llvm::omp::target::ompt::Devices;

std::atomic<uint64_t> llvm::omp::target::ompt::TracingTypesEnabled{0};

bool llvm::omp::target::ompt::TracingActive = false;

void llvm::omp::target::ompt::setTracingState(bool State) {
  TracingActive = State;
}

std::shared_ptr<llvm::sys::DynamicLibrary>
llvm::omp::target::ompt::getParentLibrary() {
  static bool ParentLibraryAssigned = false;
  if (!ParentLibraryAssigned) {
    setParentLibrary("libomptarget.so");
    ParentLibraryAssigned = true;
  }
  return ParentLibrary;
}

void llvm::omp::target::ompt::setParentLibrary(const char *Filename) {
  if (ParentLibrary)
    return;
  std::string ErrorMsg;
  ParentLibrary = std::make_shared<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(Filename, &ErrorMsg));
  if ((ParentLibrary == nullptr) || (!ParentLibrary->isValid()))
    REPORT("Failed to set parent library: %s\n", ErrorMsg.c_str());
}

int llvm::omp::target::ompt::getDeviceId(ompt_device_t *Device) {
  // Block other threads, which might trigger an erase (for the same device)
  std::unique_lock<std::mutex> Lock(DeviceIdWritingMutex);
  auto DeviceIterator = Devices.find(Device);
  if (Device == nullptr || DeviceIterator == Devices.end()) {
    REPORT("Failed to get ID for device=%p\n", Device);
    return -1;
  }
  return DeviceIterator->second;
}

void llvm::omp::target::ompt::setDeviceId(ompt_device_t *Device,
                                          int32_t DeviceId) {
  assert(Device && "Mapping device id to nullptr is not allowed");
  if (Device == nullptr) {
    REPORT("Failed to set ID for nullptr device\n");
    return;
  }
  std::unique_lock<std::mutex> Lock(DeviceIdWritingMutex);
  Devices.emplace(Device, DeviceId);
}

void llvm::omp::target::ompt::removeDeviceId(ompt_device_t *Device) {
  if (Device == nullptr) {
    REPORT("Failed to remove ID for nullptr device\n");
    return;
  }
  std::unique_lock<std::mutex> Lock(DeviceIdWritingMutex);
  Devices.erase(Device);
}

OMPT_API_ROUTINE ompt_set_result_t ompt_set_trace_ompt(ompt_device_t *Device,
                                                       unsigned int Enable,
                                                       unsigned int EventTy) {
  DP("Executing ompt_set_trace_ompt\n");

  // TODO handle device
  std::unique_lock<std::mutex> Lock(ompt_set_trace_ompt_mutex);
  ensureFuncPtrLoaded<libomptarget_ompt_set_trace_ompt_t>(
      "libomptarget_ompt_set_trace_ompt", &ompt_set_trace_ompt_fn);
  assert(ompt_set_trace_ompt_fn && "libomptarget_ompt_set_trace_ompt loaded");
  return ompt_set_trace_ompt_fn(Device, Enable, EventTy);
}

OMPT_API_ROUTINE int
ompt_start_trace(ompt_device_t *Device, ompt_callback_buffer_request_t Request,
                 ompt_callback_buffer_complete_t Complete) {
  DP("Executing ompt_start_trace\n");

  int DeviceId = getDeviceId(Device);
  {
    // Protect the function pointer
    std::unique_lock<std::mutex> Lock(ompt_start_trace_mutex);

    if (Request && Complete) {
      llvm::omp::target::ompt::setTracingState(/*Enabled=*/true);
      // Enable asynchronous memory copy profiling
      setOmptAsyncCopyProfile(/*Enable=*/true);
      // Enable queue dispatch profiling
      if (DeviceId >= 0)
        setGlobalOmptKernelProfile(DeviceId, /*Enable=*/1);
      else
        REPORT("May not enable kernel profiling for invalid device id=%d\n",
               DeviceId);
    }

    // Call libomptarget specific function
    ensureFuncPtrLoaded<libomptarget_ompt_start_trace_t>(
        "libomptarget_ompt_start_trace", &ompt_start_trace_fn);
    assert(ompt_start_trace_fn && "libomptarget_ompt_start_trace loaded");
  }
  return ompt_start_trace_fn(DeviceId, Request, Complete);
}

OMPT_API_ROUTINE int ompt_flush_trace(ompt_device_t *Device) {
  DP("Executing ompt_flush_trace\n");

  // TODO handle device
  std::unique_lock<std::mutex> Lock(ompt_flush_trace_mutex);
  ensureFuncPtrLoaded<libomptarget_ompt_flush_trace_t>(
      "libomptarget_ompt_flush_trace", &ompt_flush_trace_fn);
  assert(ompt_flush_trace_fn && "libomptarget_ompt_flush_trace loaded");
  return ompt_flush_trace_fn(getDeviceId(Device));
}

OMPT_API_ROUTINE int ompt_stop_trace(ompt_device_t *Device) {
  DP("Executing ompt_stop_trace\n");

  // TODO handle device
  {
    // Protect the function pointer
    std::unique_lock<std::mutex> Lock(ompt_stop_trace_mutex);
    llvm::omp::target::ompt::setTracingState(/*Enabled=*/false);
    // Disable asynchronous memory copy profiling
    setOmptAsyncCopyProfile(/*Enable=*/false);
    // Disable queue dispatch profiling
    int DeviceId = getDeviceId(Device);
    if (DeviceId >= 0)
      setGlobalOmptKernelProfile(DeviceId, /*Enable=*/0);
    else
      REPORT("May not disable kernel profiling for invalid device id=%d\n",
             DeviceId);
    ensureFuncPtrLoaded<libomptarget_ompt_stop_trace_t>(
        "libomptarget_ompt_stop_trace", &ompt_stop_trace_fn);
    assert(ompt_stop_trace_fn && "libomptarget_ompt_stop_trace loaded");
  }
  return ompt_stop_trace_fn(getDeviceId(Device));
}

OMPT_API_ROUTINE ompt_record_ompt_t *
ompt_get_record_ompt(ompt_buffer_t *Buffer, ompt_buffer_cursor_t CurrentPos) {
  // TODO In debug mode, get the metadata associated with this buffer
  // and assert that there are enough bytes for the current record

  // Currently, no synchronization required since a disjoint set of
  // trace records is handed over to a thread.

  // Note that CurrentPos can be nullptr. In that case, we return
  // nullptr. The tool has to handle that properly.
  return (ompt_record_ompt_t *)CurrentPos;
}

OMPT_API_ROUTINE int ompt_advance_buffer_cursor(ompt_device_t *Device,
                                                ompt_buffer_t *Buffer,
                                                size_t Size,
                                                ompt_buffer_cursor_t CurrentPos,
                                                ompt_buffer_cursor_t *NextPos) {
  // Note: The input parameter size is unused here. It refers to the
  // bytes returned in the corresponding callback.
  // Advance can be called concurrently, so synchronize setting the
  // function pointer. The actual libomptarget function does not need
  // to be synchronized since it must be working on logically disjoint
  // buffers.
  std::unique_lock<std::mutex> Lock(ompt_advance_buffer_cursor_mutex);
  ensureFuncPtrLoaded<libomptarget_ompt_advance_buffer_cursor_t>(
      "libomptarget_ompt_advance_buffer_cursor",
      &ompt_advance_buffer_cursor_fn);
  assert(ompt_advance_buffer_cursor_fn &&
         "libomptarget_ompt_advance_buffer_cursor loaded");
  return ompt_advance_buffer_cursor_fn(Device, Buffer, Size, CurrentPos,
                                       NextPos);
}

OMPT_API_ROUTINE ompt_record_t
ompt_get_record_type(ompt_buffer_t *Buffer, ompt_buffer_cursor_t CurrentPos) {
  std::unique_lock<std::mutex> Lock(ompt_get_record_type_mutex);
  ensureFuncPtrLoaded<libomptarget_ompt_get_record_type_t>(
      "libomptarget_ompt_get_record_type", &ompt_get_record_type_fn);
  assert(ompt_get_record_type_fn && "libomptarget_ompt_get_record_type loaded");
  return ompt_get_record_type_fn(Buffer, CurrentPos);
}

OMPT_API_ROUTINE ompt_device_time_t
ompt_get_device_time(ompt_device_t *Device) {
  DP("Executing ompt_get_device_time\n");
  return getSystemTimestampInNs();
}

OMPT_API_ROUTINE double ompt_translate_time(ompt_device_t *Device,
                                            ompt_device_time_t DeviceTime) {
  // Translate a device time to a meaningful timepoint in host time
  // We do not need to account for clock-skew / drift. So simple linear
  // translation using the host to device rate we obtained.
  double TranslatedTime = DeviceTime * HostToDeviceSlope + HostToDeviceOffset;
  DP("D2H translated time: %f\n", TranslatedTime);

  return TranslatedTime;
}

void llvm::omp::target::ompt::setOmptTimestamp(uint64_t StartTime,
                                               uint64_t EndTime) {
  std::unique_lock<std::mutex> Lock(ompt_set_timestamp_mutex);
  ensureFuncPtrLoaded<libomptarget_ompt_set_timestamp_t>(
      "libomptarget_ompt_set_timestamp", &ompt_set_timestamp_fn);
  // No need to hold a lock
  ompt_set_timestamp_fn(StartTime, EndTime);
}

void llvm::omp::target::ompt::setOmptHostToDeviceRate(double Slope,
                                                      double Offset) {
  HostToDeviceSlope = Slope;
  HostToDeviceOffset = Offset;
}

void llvm::omp::target::ompt::setOmptGrantedNumTeams(uint64_t NumTeams) {
  std::unique_lock<std::mutex> Lock(ompt_set_granted_teams_mutex);
  ensureFuncPtrLoaded<libomptarget_ompt_set_granted_teams_t>(
      "libomptarget_ompt_set_granted_teams", &ompt_set_granted_teams_fn);
  // No need to hold a lock
  ompt_set_granted_teams_fn(NumTeams);
}

ompt_interface_fn_t llvm::omp::target::ompt::lookupDeviceTracingFn(
    const char *InterfaceFunctionName) {
#define compareAgainst(AvailableFunction)                                      \
  if (strcmp(InterfaceFunctionName, #AvailableFunction) == 0)                  \
    return (ompt_interface_fn_t)AvailableFunction;

  FOREACH_OMPT_DEVICE_TRACING_FN(compareAgainst);
#undef compareAgainst

  DP("Warning: Could not find requested function '%s'\n",
     InterfaceFunctionName);
  return (ompt_interface_fn_t) nullptr;
}

#pragma pop_macro("DEBUG_PREFIX")

#endif // OMPT_SUPPORT
