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

#include "llvm/Support/DynamicLibrary.h"

#include <assert.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

#include "Debug.h"
#include "OmptCallback.h"
#include "OmptConnector.h"
#include "OmptDeviceTracing.h"
#include "OmptTracing.h"
#include "OmptTracingBuffer.h"
#include "omp-tools.h"

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

using namespace llvm::omp::target::ompt;

bool llvm::omp::target::ompt::TracingInitialized = false;

std::atomic<uint64_t> llvm::omp::target::ompt::TracingTypesEnabled{0};

std::shared_ptr<llvm::sys::DynamicLibrary>
    llvm::omp::target::ompt::parent_dyn_lib(nullptr);

ompt_callback_buffer_request_t
    llvm::omp::target::ompt::ompt_callback_buffer_request_fn = nullptr;
ompt_callback_buffer_complete_t
    llvm::omp::target::ompt::ompt_callback_buffer_complete_fn = nullptr;

/// ****************************************************************************
/// * ToDo: mhalk Duplicated code BELOW >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
/// ****************************************************************************

void llvm::omp::target::ompt::set_buffer_request(
    ompt_callback_buffer_request_t callback) {
  ompt_callback_buffer_request_fn = callback;
}

void llvm::omp::target::ompt::set_buffer_complete(
    ompt_callback_buffer_complete_t callback) {
  ompt_callback_buffer_complete_fn = callback;
}

void llvm::omp::target::ompt::ompt_callback_buffer_request(
    int device_num, ompt_buffer_t **buffer, size_t *bytes) {
  if (ompt_callback_buffer_request_fn) {
    ompt_callback_buffer_request_fn(device_num, buffer, bytes);
  }
}

void llvm::omp::target::ompt::ompt_callback_buffer_complete(
    int device_num, ompt_buffer_t *buffer, size_t bytes,
    ompt_buffer_cursor_t begin, int buffer_owned) {
  if (ompt_callback_buffer_complete_fn) {
    ompt_callback_buffer_complete_fn(device_num, buffer, bytes, begin,
                                     buffer_owned);
  }
}

void llvm::omp::target::ompt::set_tracing_state(bool State) {
  TracingInitialized = State;
}

bool llvm::omp::target::ompt::is_tracing_type_enabled(unsigned int etype) {
  assert(etype < 64);
  if (etype < 64)
    return (TracingTypesEnabled & (1UL << etype)) != 0;
  return false;
}

void llvm::omp::target::ompt::set_tracing_type_enabled(unsigned int etype,
                                                       bool b) {
  assert(etype < 64);
  if (etype < 64) {
    if (b)
      TracingTypesEnabled |= (1UL << etype);
    else
      TracingTypesEnabled &= ~(1UL << etype);
  }
}

ompt_set_result_t llvm::omp::target::ompt::set_trace_ompt(ompt_device_t *device,
                                                          unsigned int enable,
                                                          unsigned int etype) {
  // TODO handle device

  DP("set_trace_ompt: %d %d\n", etype, enable);

  bool is_event_enabled = enable > 0;
  if (etype == 0) {
    // set/reset all supported types
    set_tracing_type_enabled(ompt_callbacks_t::ompt_callback_target,
                             is_event_enabled);
    set_tracing_type_enabled(ompt_callbacks_t::ompt_callback_target_data_op,
                             is_event_enabled);
    set_tracing_type_enabled(ompt_callbacks_t::ompt_callback_target_submit,
                             is_event_enabled);
    set_tracing_type_enabled(ompt_callbacks_t::ompt_callback_target_emi,
                             is_event_enabled);
    set_tracing_type_enabled(ompt_callbacks_t::ompt_callback_target_data_op_emi,
                             is_event_enabled);
    set_tracing_type_enabled(ompt_callbacks_t::ompt_callback_target_submit_emi,
                             is_event_enabled);

    if (is_event_enabled)
      return ompt_set_sometimes; // a subset is enabled
    else
      return ompt_set_always; // we can disable for all
  }
  switch (etype) {
  case ompt_callbacks_t::ompt_callback_target:
  case ompt_callbacks_t::ompt_callback_target_data_op:
  case ompt_callbacks_t::ompt_callback_target_submit:
  case ompt_callbacks_t::ompt_callback_target_emi:
  case ompt_callbacks_t::ompt_callback_target_data_op_emi:
  case ompt_callbacks_t::ompt_callback_target_submit_emi: {
    set_tracing_type_enabled(etype, is_event_enabled);
    return ompt_set_always;
  }
  default: {
    if (is_event_enabled)
      return ompt_set_never; // unimplemented
    else
      return ompt_set_always; // always disabled anyways
  }
  }
}

/// ****************************************************************************
/// * ToDo: mhalk Duplicated code ABOVE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
/// ****************************************************************************

void llvm::omp::target::ompt::compute_parent_dyn_lib(const char *lib_name) {
  if (parent_dyn_lib)
    return;
  std::string err_msg;
  parent_dyn_lib = std::make_shared<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(lib_name, &err_msg));
}

std::shared_ptr<llvm::sys::DynamicLibrary>
llvm::omp::target::ompt::get_parent_dyn_lib() {
  static bool ParentLibDetermined = false;
  if (!ParentLibDetermined) {
    compute_parent_dyn_lib("libomptarget.so");
    ParentLibDetermined = true;
  }
  return parent_dyn_lib;
}

OMPT_API_ROUTINE ompt_set_result_t ompt_set_trace_ompt(ompt_device_t *device,
                                                       unsigned int enable,
                                                       unsigned int etype) {
  DP("Executing ompt_set_trace_ompt\n");

  // TODO handle device
  std::unique_lock<std::mutex> L(set_trace_mutex);
  llvm::omp::target::ompt::set_trace_ompt(device, enable, etype);
  ensureFuncPtrLoaded<libomptarget_ompt_set_trace_ompt_t>(
      "libomptarget_ompt_set_trace_ompt", &ompt_set_trace_ompt_fn);
  assert(ompt_set_trace_ompt_fn && "libomptarget_ompt_set_trace_ompt loaded");
  return ompt_set_trace_ompt_fn(device, enable, etype);
}

OMPT_API_ROUTINE int
ompt_start_trace(ompt_device_t *device, ompt_callback_buffer_request_t request,
                 ompt_callback_buffer_complete_t complete) {
  DP("OMPT: Executing ompt_start_trace\n");

  // TODO handle device

  {
    // protect the function pointer
    std::unique_lock<std::mutex> lck(start_trace_mutex);
    // plugin specific
    llvm::omp::target::ompt::set_buffer_request(request);
    llvm::omp::target::ompt::set_buffer_complete(complete);
    if (request && complete) {
      llvm::omp::target::ompt::set_tracing_state(/*Enabled=*/true);
      // Enable asynchronous memory copy profiling
      setOmptAsyncCopyProfile(/*Enable=*/true);
      // Enable queue dispatch profiling
      // ToDo: mhalk provide device id / number here
      setGlobalOmptKernelProfile(/*DeviceId=*/0, /*Enable=*/1);
    }

    // libomptarget specific
    ensureFuncPtrLoaded<libomptarget_ompt_start_trace_t>(
        "libomptarget_ompt_start_trace", &ompt_start_trace_fn);
    assert(ompt_start_trace_fn && "libomptarget_ompt_start_trace loaded");
  }

  return ompt_start_trace_fn(request, complete);
}

OMPT_API_ROUTINE int ompt_flush_trace(ompt_device_t *device) {
  DP("OMPT: Executing ompt_flush_trace\n");

  // TODO handle device
  std::unique_lock<std::mutex> L(flush_trace_mutex);
  ensureFuncPtrLoaded<libomptarget_ompt_flush_trace_t>(
      "libomptarget_ompt_flush_trace", &ompt_flush_trace_fn);
  assert(ompt_start_trace_fn && "libomptarget_ompt_flush_trace loaded");
  return ompt_flush_trace_fn(device);
}

OMPT_API_ROUTINE int ompt_stop_trace(ompt_device_t *device) {
  DP("OMPT: Executing ompt_stop_trace\n");

  // TODO handle device
  {
    // Protect the function pointer
    std::unique_lock<std::mutex> lck(stop_trace_mutex);
    llvm::omp::target::ompt::set_tracing_state(/*Enabled=*/false);
    // Disable asynchronous memory copy profiling
    setOmptAsyncCopyProfile(/*Enable=*/false);
    // Disable queue dispatch profiling
    setGlobalOmptKernelProfile(0, /*Enable=*/0);
    ensureFuncPtrLoaded<libomptarget_ompt_stop_trace_t>(
        "libomptarget_ompt_stop_trace", &ompt_stop_trace_fn);
    assert(ompt_stop_trace_fn && "libomptarget_ompt_stop_trace loaded");
  }

  return ompt_stop_trace_fn(device);
}

OMPT_API_ROUTINE ompt_record_ompt_t *
ompt_get_record_ompt(ompt_buffer_t *buffer, ompt_buffer_cursor_t current) {
  // TODO In debug mode, get the metadata associated with this buffer
  // and assert that there are enough bytes for the current record

  // Currently, no synchronization required since a disjoint set of
  // trace records is handed over to a thread.

  // Note that current can be nullptr. In that case, we return
  // nullptr. The tool has to handle that properly.
  return (ompt_record_ompt_t *)current;
}

OMPT_API_ROUTINE int
ompt_advance_buffer_cursor(ompt_device_t *device, ompt_buffer_t *buffer,
                           size_t size, /* bytes returned in the corresponding
                                           callback, unused here */
                           ompt_buffer_cursor_t current,
                           ompt_buffer_cursor_t *next) {
  // Advance can be called concurrently, so synchronize setting the
  // function pointer. The actual libomptarget function does not need
  // to be synchronized since it must be working on logically disjoint
  // buffers.
  std::unique_lock<std::mutex> L(advance_buffer_cursor_mutex);
  ensureFuncPtrLoaded<libomptarget_ompt_advance_buffer_cursor_t>(
      "libomptarget_ompt_advance_buffer_cursor",
      &ompt_advance_buffer_cursor_fn);
  assert(ompt_advance_buffer_cursor_fn &&
         "libomptarget_ompt_advance_buffer_cursor loaded");
  return ompt_advance_buffer_cursor_fn(device, buffer, size, current, next);
}

OMPT_API_ROUTINE ompt_record_t
ompt_get_record_type(ompt_buffer_t *buffer, ompt_buffer_cursor_t current) {
  std::unique_lock<std::mutex> L(get_record_type_mutex);
  ensureFuncPtrLoaded<libomptarget_ompt_get_record_type_t>(
      "libomptarget_ompt_get_record_type", &ompt_get_record_type_fn);
  assert(ompt_get_record_type_fn && "libomptarget_ompt_get_record_type loaded");
  return ompt_get_record_type_fn(buffer, current);
}

OMPT_API_ROUTINE ompt_device_time_t
ompt_get_device_time(ompt_device_t *device) {
  DP("OMPT: Executing ompt_get_device_time\n");
  return getSystemTimestampInNs();
}

// Translates a device time to a meaningful timepoint in host time
OMPT_API_ROUTINE double ompt_translate_time(ompt_device_t *device,
                                            ompt_device_time_t device_time) {
  // We do not need to account for clock-skew / drift. So simple linear
  // translation using the host to device rate we obtained.
  double TranslatedTime = device_time * HostToDeviceSlope + HostToDeviceOffset;
  DP("OMPT: Translate time: %f\n", TranslatedTime);

  return TranslatedTime;
}

/// Set timestamps in trace records.
void setOmptTimestamp(uint64_t StartTime, uint64_t EndTime) {
  std::unique_lock<std::mutex> L(ompt_set_timestamp_mtx);
  ensureFuncPtrLoaded<libomptarget_ompt_set_timestamp_t>(
      "libomptarget_ompt_set_timestamp", &ompt_set_timestamp_fn);
  // No need to hold a lock
  ompt_set_timestamp_fn(StartTime, EndTime);
}

void setOmptHostToDeviceRate(double Slope, double Offset) {
  HostToDeviceSlope = Slope;
  HostToDeviceOffset = Offset;
}

/// Set granted number of teams in trace records.
void setOmptGrantedNumTeams(uint64_t NumTeams) {
  std::unique_lock<std::mutex> L(granted_teams_mtx);
  ensureFuncPtrLoaded<libomptarget_ompt_set_granted_teams_t>(
      "libomptarget_ompt_set_granted_teams", &ompt_set_granted_teams_fn);
  // No need to hold a lock
  ompt_set_granted_teams_fn(NumTeams);
}

ompt_interface_fn_t
llvm::omp::target::ompt::doLookup(const char *InterfaceFunctionName) {
#define macro(fn)                                                              \
  if (strcmp(InterfaceFunctionName, #fn) == 0)                                 \
    return (ompt_interface_fn_t)fn;

  FOREACH_OMPT_DEVICE_TRACING_FN(macro);

#undef macro
  return (ompt_interface_fn_t)lookupCallbackByName(InterfaceFunctionName);
}

#pragma pop_macro("DEBUG_PREFIX")

#endif // OMPT_SUPPORT
