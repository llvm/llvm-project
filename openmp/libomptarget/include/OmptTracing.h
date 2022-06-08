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

#ifndef OPENMP_LIBOMPTARGET_OMPTTRACING_H
#define OPENMP_LIBOMPTARGET_OMPTTRACING_H

#ifdef OMPT_SUPPORT

#include "OmptCommon.h"

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

namespace llvm {
namespace omp {
namespace target {
namespace ompt {

enum class TraceOperations { SET, START, STOP, FLUSH, SELECT_NEXT, GET_TYPE };

#define declareOmptTracingFunction(Name)                                       \
  extern libomptarget_##Name##_t Name##_fn;
FOREACH_OMPT_DEVICE_TRACING_FN(declareOmptTracingFunction)
#undef declareOmptTracingFunction

void ompt_callback_buffer_request(int device_num, ompt_buffer_t **buffer,
                                  size_t *bytes);

void ompt_callback_buffer_complete(int device_num, ompt_buffer_t *buffer,
                                   size_t bytes, ompt_buffer_cursor_t begin,
                                   int buffer_owned);

void set_buffer_request(ompt_callback_buffer_request_t callback);

void set_buffer_complete(ompt_callback_buffer_complete_t callback);

void set_tracing_state(bool Enabled);

bool is_tracing_type_enabled(unsigned int etype);

void set_tracing_type_enabled(unsigned int etype, bool b);

ompt_set_result_t set_trace_ompt(ompt_device_t *device, unsigned int enable,
                                 unsigned int etype);

extern std::shared_ptr<llvm::sys::DynamicLibrary> parent_dyn_lib;

extern ompt_callback_buffer_request_t ompt_callback_buffer_request_fn;
extern ompt_callback_buffer_complete_t ompt_callback_buffer_complete_fn;

// Mutexes to serialize invocation of device-independent entry points
extern std::mutex TraceAccessMutex;
extern std::mutex TraceControlMutex;

// Serialize calls to std::hash
// thread_id_hash_mutex
extern std::mutex TraceHashThreadMutex;

// Mutexes to protect the function pointers
extern std::mutex set_trace_mutex;
extern std::mutex start_trace_mutex;
extern std::mutex flush_trace_mutex;
extern std::mutex stop_trace_mutex;
extern std::mutex advance_buffer_cursor_mutex;
extern std::mutex get_record_type_mutex;

// Thread local variables used by the plugin to communicate OMPT information
// that are then used to populate trace records. This method assumes a
// synchronous implementation, otherwise it won't work.

// static thread_local ompt_num_granted_teams
extern thread_local uint32_t TraceRecordNumGrantedTeams;

// static thread_local ompt_tr_start_time
extern thread_local uint64_t TraceRecordStartTime;

// static thread_local ompt_tr_end_time
extern thread_local uint64_t TraceRecordStopTime;

extern std::atomic<uint64_t> TracingTypesEnabled;

/// OMPT tracing initialization status; false if initializeLibrary has not been
/// executed
extern bool TracingInitialized;

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm

#pragma pop_macro("DEBUG_PREFIX")

#endif // OMPT_SUPPORT

#endif // OPENMP_LIBOMPTARGET_OMPTTRACING_H
