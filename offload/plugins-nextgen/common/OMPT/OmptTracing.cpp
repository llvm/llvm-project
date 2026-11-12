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

#include "OmptDeviceTracing.h"
#include "Shared/Debug.h"
#include "omp-tools.h"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <mutex>

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

// Define OMPT device tracing function mutexes
#define defineOmptTracingFnMutex(Name)                                         \
  std::mutex llvm::omp::target::ompt::Name##_mutex;
FOREACH_OMPT_DEVICE_TRACING_FN_IMPLEMENTAIONS(defineOmptTracingFnMutex)
#undef defineOmptTracingFnMutex

std::mutex llvm::omp::target::ompt::DeviceIdWritingMutex;

using namespace llvm::omp::target::ompt;
using namespace llvm::omp::target::debug;

double llvm::omp::target::ompt::HostToDeviceSlope = .0;
double llvm::omp::target::ompt::HostToDeviceOffset = .0;

std::map<ompt_device_t *, int32_t> llvm::omp::target::ompt::Devices;

int llvm::omp::target::ompt::getDeviceId(ompt_device_t *Device) {
  // Block other threads, which might trigger an erase (for the same device)
  std::unique_lock<std::mutex> Lock(DeviceIdWritingMutex);
  auto DeviceIterator = Devices.find(Device);
  if (Device == nullptr || DeviceIterator == Devices.end()) {
    REPORT() << "Failed to get ID for Device=" << Device;
    return -1;
  }
  return DeviceIterator->second;
}

void llvm::omp::target::ompt::setDeviceId(ompt_device_t *Device,
                                          int32_t DeviceId) {
  assert(Device && "Mapping device ID to nullptr is not allowed");
  if (Device == nullptr || DeviceId < 0) {
    REPORT() << "Failed to set ID=%d for Device=" << DeviceId << Device;
    return;
  }
  std::unique_lock<std::mutex> Lock(DeviceIdWritingMutex);
  auto DeviceIterator = Devices.find(Device);
  if (DeviceIterator != Devices.end()) {
    auto CurrentDeviceId = DeviceIterator->second;
    if (DeviceId == CurrentDeviceId) {
      REPORT() << "Tried to duplicate OMPT Device= " << Device
               << " ID=" << DeviceId;
    } else {
      REPORT() << "Tried to overwrite OMPT Device=" << Device
               << " (ID=" << CurrentDeviceId << " with new ID=" << DeviceId;
    }
    return;
  }
  Devices.emplace(Device, DeviceId);
}

void llvm::omp::target::ompt::removeDeviceId(ompt_device_t *Device) {
  int DeviceId = getDeviceId(Device);
  if (DeviceId < 0) {
    REPORT() << "Tried to remove Device= " << Device << " ID=" << DeviceId;
    return;
  }
  std::unique_lock<std::mutex> Lock(DeviceIdWritingMutex);
  Devices.erase(Device);
  TracedDevices.erase(DeviceId);
}

OMPT_API_ROUTINE ompt_set_result_t ompt_set_trace_ompt(ompt_device_t *Device,
                                                       unsigned int Enable,
                                                       unsigned int EventTy) {
  ODBG(ODT_Tool) << "Executing ompt_set_trace_ompt";

  int DeviceId = getDeviceId(Device);
  if (DeviceId < 0) {
    REPORT() << "Failed to set trace events for Device=" << Device
             << " (Unknown device) [Enable=" << Enable
             << " EventTy=" << EventTy;
    return ompt_set_never;
  }

  std::unique_lock<std::mutex> Lock(ompt_set_trace_ompt_mutex);
  return libomptarget_ompt_set_trace_ompt(DeviceId, Enable, EventTy);
}

OMPT_API_ROUTINE int
ompt_start_trace(ompt_device_t *Device, ompt_callback_buffer_request_t Request,
                 ompt_callback_buffer_complete_t Complete) {
  ODBG(ODT_Tool) << "Executing ompt_start_trace";

  int DeviceId = getDeviceId(Device);
  if (DeviceId < 0) {
    REPORT() << "Failed to start trace for Device=" << Device
             << " (Unknown device";
    // Indicate failure
    return 0;
  }

  {
    // Serialize enabling the device-side profiling sources
    std::unique_lock<std::mutex> Lock(ompt_start_trace_mutex);

    if (Request && Complete) {
      llvm::omp::target::ompt::enableDeviceTracing(DeviceId);
      // Enable asynchronous memory copy profiling
      setOmptAsyncCopyProfile(/*Enable=*/true);
      // Enable queue dispatch profiling
      if (DeviceId >= 0)
        setGlobalOmptKernelProfile(Device, /*Enable=*/1);
      else
        REPORT() << "May not enable kernel profiling for invalid device id="
                 << DeviceId;
    }
  }
  return libomptarget_ompt_start_trace(DeviceId, Request, Complete);
}

OMPT_API_ROUTINE int ompt_flush_trace(ompt_device_t *Device) {
  ODBG(ODT_Tool) << "Executing ompt_flush_trace";

  std::unique_lock<std::mutex> Lock(ompt_flush_trace_mutex);
  return libomptarget_ompt_flush_trace(getDeviceId(Device));
}

OMPT_API_ROUTINE int ompt_stop_trace(ompt_device_t *Device) {
  ODBG(ODT_Tool) << "Executing ompt_stop_trace";

  int DeviceId = getDeviceId(Device);
  if (DeviceId < 0) {
    REPORT() << "Failed to stop trace for Device=" << Device
             << " (Unknown device)";
    // Indicate failure
    return 0;
  }

  {
    // Serialize disabling the device-side profiling sources
    std::unique_lock<std::mutex> Lock(ompt_stop_trace_mutex);
    llvm::omp::target::ompt::disableDeviceTracing(DeviceId);
    // Disable asynchronous memory copy profiling
    setOmptAsyncCopyProfile(/*Enable=*/false);
    // Disable queue dispatch profiling
    if (DeviceId >= 0)
      setGlobalOmptKernelProfile(Device, /*Enable=*/0);
    else
      REPORT() << "May not disable kernel profiling for invalid device id="
               << DeviceId;
  }
  return libomptarget_ompt_stop_trace(DeviceId);
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
  // The libomptarget function does not need to be synchronized since it must
  // be working on logically disjoint buffers.
  std::unique_lock<std::mutex> Lock(ompt_advance_buffer_cursor_mutex);
  return libomptarget_ompt_advance_buffer_cursor(Device, Buffer, Size,
                                                 CurrentPos, NextPos);
}

OMPT_API_ROUTINE ompt_record_t
ompt_get_record_type(ompt_buffer_t *Buffer, ompt_buffer_cursor_t CurrentPos) {
  std::unique_lock<std::mutex> Lock(ompt_get_record_type_mutex);
  return libomptarget_ompt_get_record_type(Buffer, CurrentPos);
}

OMPT_API_ROUTINE ompt_device_time_t
ompt_get_device_time(ompt_device_t *Device) {
  ODBG(ODT_Tool) << "Executing ompt_get_device_time";
  return getSystemTimestampInNs();
}

OMPT_API_ROUTINE double ompt_translate_time(ompt_device_t *Device,
                                            ompt_device_time_t DeviceTime) {
  // Translate a device time to a meaningful timepoint in host time
  // We do not need to account for clock-skew / drift. So simple linear
  // translation using the host to device rate we obtained.
  double TranslatedTime = DeviceTime * HostToDeviceSlope + HostToDeviceOffset;
  ODBG(ODT_Tool) << "D2H translated time: " << TranslatedTime;

  return TranslatedTime;
}

void llvm::omp::target::ompt::setOmptTimestamp(uint64_t StartTime,
                                               uint64_t EndTime) {
  std::unique_lock<std::mutex> Lock(ompt_set_timestamp_mutex);
  libomptarget_ompt_set_timestamp(StartTime, EndTime);
}

void llvm::omp::target::ompt::setOmptHostToDeviceRate(double Slope,
                                                      double Offset) {
  HostToDeviceSlope = Slope;
  HostToDeviceOffset = Offset;
}

void llvm::omp::target::ompt::setOmptGrantedNumTeams(uint64_t NumTeams) {
  std::unique_lock<std::mutex> Lock(ompt_set_granted_teams_mutex);
  libomptarget_ompt_set_granted_teams(NumTeams);
}

ompt_interface_fn_t llvm::omp::target::ompt::lookupDeviceTracingFn(
    const char *InterfaceFunctionName) {
#define compareAgainst(AvailableFunction)                                      \
  if (strcmp(InterfaceFunctionName, #AvailableFunction) == 0)                  \
    return (ompt_interface_fn_t)AvailableFunction;

  FOREACH_OMPT_DEVICE_TRACING_FN(compareAgainst);
#undef compareAgainst

  ODBG(ODT_Tool) << "Warning: Could not find requested function "
                 << InterfaceFunctionName;
  return (ompt_interface_fn_t) nullptr;
}

#pragma pop_macro("DEBUG_PREFIX")

#endif // OMPT_SUPPORT
