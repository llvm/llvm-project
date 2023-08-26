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

#include "llvm/Support/DynamicLibrary.h"

#include <assert.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#include "Debug.h"
#include "OmptCallback.h"
#include "OmptConnector.h"
#include "OmptInterface.h"
#include "OmptTracing.h"
#include "OmptTracingBuffer.h"
#include "omp-tools.h"
#include "private.h"

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

using namespace llvm::omp::target::ompt;

// ToDo: mhalk
OmptTracingBufferMgr ompt_trace_record_buffer_mgr;

// ToDo: mhalk check storage specifier for mutexes
std::mutex llvm::omp::target::ompt::TraceAccessMutex;
std::mutex llvm::omp::target::ompt::TraceControlMutex;
std::mutex llvm::omp::target::ompt::TraceHashThreadMutex;

// ToDo: mhalk check storage specifier for TraceRecord...
thread_local uint32_t llvm::omp::target::ompt::TraceRecordNumGrantedTeams = 0;
thread_local uint64_t llvm::omp::target::ompt::TraceRecordStartTime = 0;
thread_local uint64_t llvm::omp::target::ompt::TraceRecordStopTime = 0;

bool llvm::omp::target::ompt::TracingInitialized = false;

std::atomic<uint64_t> llvm::omp::target::ompt::TracingTypesEnabled{0};

/// ****************************************************************************
/// * ToDo: mhalk Duplicated code BELOW >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
/// ****************************************************************************

std::shared_ptr<llvm::sys::DynamicLibrary>
    llvm::omp::target::ompt::parent_dyn_lib(nullptr);

ompt_callback_buffer_request_t
    llvm::omp::target::ompt::ompt_callback_buffer_request_fn = nullptr;
ompt_callback_buffer_complete_t
    llvm::omp::target::ompt::ompt_callback_buffer_complete_fn = nullptr;

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

ompt_record_ompt_t *Interface::target_data_submit_trace_record_gen(
    ompt_target_data_op_t data_op, void *src_addr, int64_t src_device_num,
    void *dest_addr, int64_t dest_device_num, size_t bytes) {
  if (!llvm::omp::target::ompt::TracingInitialized ||
      (!llvm::omp::target::ompt::is_tracing_type_enabled(
           ompt_callback_target_data_op) &&
       !llvm::omp::target::ompt::is_tracing_type_enabled(
           ompt_callback_target_data_op_emi)))
    return nullptr;

  ompt_record_ompt_t *data_ptr =
      (ompt_record_ompt_t *)ompt_trace_record_buffer_mgr.assignCursor(
          ompt_callback_target_data_op);

  // Logically, this record is now private

  set_trace_record_common(data_ptr, ompt_callback_target_data_op);

  set_trace_record_target_data_op(&data_ptr->record.target_data_op, data_op,
                                  src_addr, src_device_num, dest_addr,
                                  dest_device_num, bytes);

  // The trace record has been created, mark it ready for delivery to the tool
  ompt_trace_record_buffer_mgr.setTRStatus(data_ptr,
                                           OmptTracingBufferMgr::TR_ready);

  DP("Generated target_data_submit trace record %p\n", data_ptr);
  return data_ptr;
}

void Interface::set_trace_record_target_data_op(
    ompt_record_target_data_op_t *rec, ompt_target_data_op_t data_op,
    void *src_addr, int64_t src_device_num, void *dest_addr,
    int64_t dest_device_num, size_t bytes) {
  rec->host_op_id = HostOpId;
  rec->optype = data_op;
  rec->src_addr = src_addr;
  rec->src_device_num = src_device_num;
  rec->dest_addr = dest_addr;
  rec->dest_device_num = dest_device_num;
  rec->bytes = bytes;
  rec->end_time = TraceRecordStopTime;
  rec->codeptr_ra = _codeptr_ra;
}

ompt_record_ompt_t *
Interface::target_submit_trace_record_gen(unsigned int num_teams) {
  if (!llvm::omp::target::ompt::TracingInitialized ||
      (!llvm::omp::target::ompt::is_tracing_type_enabled(
           ompt_callback_target_submit) &&
       !llvm::omp::target::ompt::is_tracing_type_enabled(
           ompt_callback_target_submit_emi)))
    return nullptr;

  ompt_record_ompt_t *data_ptr =
      (ompt_record_ompt_t *)ompt_trace_record_buffer_mgr.assignCursor(
          ompt_callback_target_submit);

  // Logically, this record is now private

  set_trace_record_common(data_ptr, ompt_callback_target_submit);

  set_trace_record_target_kernel(&data_ptr->record.target_kernel, num_teams);

  // The trace record has been created, mark it ready for delivery to the tool
  ompt_trace_record_buffer_mgr.setTRStatus(data_ptr,
                                           OmptTracingBufferMgr::TR_ready);

  DP("Generated target_submit trace record %p\n", data_ptr);
  return data_ptr;
}

void Interface::set_trace_record_target_kernel(ompt_record_target_kernel_t *rec,
                                               unsigned int num_teams) {
  rec->host_op_id = HostOpId;
  rec->requested_num_teams = num_teams;
  rec->granted_num_teams = TraceRecordNumGrantedTeams;
  rec->end_time = TraceRecordStopTime;
}

ompt_record_ompt_t *
Interface::target_trace_record_gen(int64_t device_id, ompt_target_t kind,
                                   ompt_scope_endpoint_t endpoint, void *code) {
  if (!llvm::omp::target::ompt::TracingInitialized ||
      (!llvm::omp::target::ompt::is_tracing_type_enabled(
           ompt_callback_target) &&
       !llvm::omp::target::ompt::is_tracing_type_enabled(
           ompt_callback_target_emi)))
    return nullptr;

  ompt_record_ompt_t *data_ptr =
      (ompt_record_ompt_t *)ompt_trace_record_buffer_mgr.assignCursor(
          ompt_callback_target);

  // Logically, this record is now private

  set_trace_record_common(data_ptr, ompt_callback_target);
  set_trace_record_target(&data_ptr->record.target, device_id, kind, endpoint,
                          code);

  // The trace record has been created, mark it ready for delivery to the tool
  ompt_trace_record_buffer_mgr.setTRStatus(data_ptr,
                                           OmptTracingBufferMgr::TR_ready);

  DP("Generated target trace record %p, completing a kernel\n", data_ptr);

  return data_ptr;
}

void Interface::set_trace_record_target(ompt_record_target_t *rec,
                                        int64_t device_id, ompt_target_t kind,
                                        ompt_scope_endpoint_t endpoint,
                                        void *code) {
  rec->kind = kind;
  rec->endpoint = endpoint;
  rec->device_num = device_id;
  assert(TaskData);
  rec->task_id = TaskData->value;
  rec->target_id = TargetData.value;
  rec->codeptr_ra = code;
}

void Interface::set_trace_record_common(ompt_record_ompt_t *data_ptr,
                                        ompt_callbacks_t cbt) {
  data_ptr->type = cbt;
  if (cbt == ompt_callback_target)
    data_ptr->time = 0; // Currently, no consumer, so no need to set it
  else
    data_ptr->time = TraceRecordStartTime;
  {
    std::unique_lock<std::mutex> lck(TraceHashThreadMutex);
    data_ptr->thread_id =
        std::hash<std::thread::id>()(std::this_thread::get_id());
  }
  data_ptr->target_id = TargetData.value;
}

extern "C" {
// Device-independent entry point for ompt_set_trace_ompt
ompt_set_result_t libomptarget_ompt_set_trace_ompt(ompt_device_t *device,
                                                   unsigned int enable,
                                                   unsigned int etype) {
  std::unique_lock<std::mutex> lck(TraceAccessMutex);
  return llvm::omp::target::ompt::set_trace_ompt(device, enable, etype);
}

// Device-independent entry point for ompt_start_trace
int libomptarget_ompt_start_trace(ompt_callback_buffer_request_t request,
                                  ompt_callback_buffer_complete_t complete) {
  std::unique_lock<std::mutex> lck(TraceControlMutex);
  llvm::omp::target::ompt::set_buffer_request(request);
  llvm::omp::target::ompt::set_buffer_complete(complete);
  if (request && complete) {
    llvm::omp::target::ompt::set_tracing_state(/*Enabled=*/true);
    ompt_trace_record_buffer_mgr.startHelperThreads();
    return 1; // success
  }
  return 0; // failure
}

// Device-independent entry point for ompt_flush_trace
int libomptarget_ompt_flush_trace(ompt_device_t *device) {
  std::unique_lock<std::mutex> lck(TraceControlMutex);
  return ompt_trace_record_buffer_mgr.flushAllBuffers(device);
}

// Device independent entry point for ompt_stop_trace
int libomptarget_ompt_stop_trace(ompt_device_t *device) {
  std::unique_lock<std::mutex> lck(TraceControlMutex);
  int status = ompt_trace_record_buffer_mgr.flushAllBuffers(device);
  // TODO shutdown should perhaps return a status
  ompt_trace_record_buffer_mgr.shutdownHelperThreads();
  llvm::omp::target::ompt::set_tracing_state(/*Enabled=*/false);
  return status;
}

// Device independent entry point for ompt_advance_buffer_cursor
// Note: The input parameter size is unused here. It refers to the
// bytes returned in the corresponding callback.
int libomptarget_ompt_advance_buffer_cursor(ompt_device_t *device,
                                            ompt_buffer_t *buffer, size_t size,
                                            ompt_buffer_cursor_t current,
                                            ompt_buffer_cursor_t *next) {
  char *curr_rec = (char *)current;
  // Don't assert if current is null, just indicate end of buffer
  if (curr_rec == nullptr ||
      ompt_trace_record_buffer_mgr.isLastCursor(curr_rec)) {
    *next = 0;
    return false;
  }
  // TODO In debug mode, assert that the metadata points to the
  // input parameter buffer

  size_t sz = ompt_trace_record_buffer_mgr.getTRSize();
  *next = (ompt_buffer_cursor_t)(curr_rec + sz);
  DP("Advanced buffer pointer by %lu bytes to %p\n", sz, curr_rec + sz);
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
ompt_record_t libomptarget_ompt_get_record_type(ompt_buffer_t *buffer,
                                                ompt_buffer_cursor_t current) {
  // TODO: When different OMPT trace buffer formats supported, this needs to be
  // fixed.
  return ompt_record_t::ompt_record_ompt;
}
} // extern "C"

#pragma pop_macro("DEBUG_PREFIX")

#else
extern "C" {
/// ToDo: mhalk Check if we need dummy definitions when OMPT is disabled
/// Example: libomptarget_ompt_set_trace_ompt
}
#endif // OMPT_SUPPORT
