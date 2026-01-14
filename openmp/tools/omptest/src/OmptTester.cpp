//===- OmptTester.cpp - ompTest OMPT tool implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file represents the core implementation file for the ompTest library.
/// It provides the actual OMPT tool implementation: registers callbacks, etc.
/// OMPT callbacks are passed to their corresponding handler, which in turn
/// notifies all registered asserters.
///
//===----------------------------------------------------------------------===//

#include "OmptTester.h"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>

using namespace omptest;

// Callback handler, which receives and relays OMPT callbacks
extern OmptCallbackHandler *Handler;

// EventListener, which actually prints the OMPT events
static OmptEventReporter *EventReporter;

// From openmp/runtime/test/ompt/callback.h
#define register_ompt_callback_t(name, type)                                   \
  do {                                                                         \
    type f_##name = &on_##name;                                                \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_ompt_callback(name) register_ompt_callback_t(name, name##_t)

#define OMPT_BUFFER_REQUEST_SIZE 256

#ifdef OPENMP_LIBOMPTEST_BUILD_STANDALONE
std::vector<std::pair<std::string, TestSuite>> TestRegistrar::Tests;
#endif

static std::atomic<ompt_id_t> NextOpId{0x8000000000000001};
static bool UseEMICallbacks = false;
static bool UseTracing = false;
static bool RunAsTestSuite = false;
static bool ColoredLog = false;

// OMPT entry point handles
static ompt_set_trace_ompt_t ompt_set_trace_ompt = 0;
static ompt_start_trace_t ompt_start_trace = 0;
static ompt_flush_trace_t ompt_flush_trace = 0;
static ompt_stop_trace_t ompt_stop_trace = 0;
static ompt_get_record_ompt_t ompt_get_record_ompt = 0;
static ompt_advance_buffer_cursor_t ompt_advance_buffer_cursor = 0;
static ompt_get_record_type_t ompt_get_record_type_fn = 0;

// OMPT device side tracing: Currently traced devices
typedef std::unordered_set<ompt_device_t *> OmptDeviceSetTy;
typedef std::unique_ptr<OmptDeviceSetTy> OmptDeviceSetPtrTy;
static OmptDeviceSetPtrTy TracedDevices;

// OMPT callbacks

// Trace record callbacks
static void on_ompt_callback_buffer_request(int device_num,
                                            ompt_buffer_t **buffer,
                                            size_t *bytes) {
  *bytes = OMPT_BUFFER_REQUEST_SIZE;
  *buffer = malloc(*bytes);
  OmptCallbackHandler::get().handleBufferRequest(device_num, buffer, bytes);
}

// Note: This callback must handle a null begin cursor. Currently,
// ompt_get_record_ompt, print_record_ompt, and
// ompt_advance_buffer_cursor handle a null cursor.
static void on_ompt_callback_buffer_complete(
    int device_num, ompt_buffer_t *buffer,
    size_t bytes, /* bytes returned in this callback */
    ompt_buffer_cursor_t begin, int buffer_owned) {
  OmptCallbackHandler::get().handleBufferComplete(device_num, buffer, bytes,
                                                  begin, buffer_owned);

  int Status = 1;
  ompt_buffer_cursor_t CurrentPos = begin;
  while (Status) {
    ompt_record_ompt_t *Record = ompt_get_record_ompt(buffer, CurrentPos);
    if (ompt_get_record_type_fn(buffer, CurrentPos) != ompt_record_ompt) {
      printf("Warning: received non-ompt type buffer object\n");
    }
    // TODO: Sometimes it may happen that the retrieved record may be null?!
    // Only handle non-null records
    if (Record != nullptr)
      OmptCallbackHandler::get().handleBufferRecord(Record);
    Status = ompt_advance_buffer_cursor(/*device=*/NULL, buffer, bytes,
                                        CurrentPos, &CurrentPos);
  }
  if (buffer_owned) {
    OmptCallbackHandler::get().handleBufferRecordDeallocation(buffer);
    free(buffer);
  }
}

static ompt_set_result_t set_trace_ompt(ompt_device_t *Device) {
  if (!ompt_set_trace_ompt)
    return ompt_set_error;

  if (UseEMICallbacks) {
    ompt_set_trace_ompt(Device, /*enable=*/1,
                        /*etype=*/ompt_callback_target_emi);
    ompt_set_trace_ompt(Device, /*enable=*/1,
                        /*etype=*/ompt_callback_target_data_op_emi);
    ompt_set_trace_ompt(Device, /*enable=*/1,
                        /*etype=*/ompt_callback_target_submit_emi);
  } else {
    ompt_set_trace_ompt(Device, /*enable=*/1, /*etype=*/ompt_callback_target);
    ompt_set_trace_ompt(Device, /*enable=*/1,
                        /*etype=*/ompt_callback_target_data_op);
    ompt_set_trace_ompt(Device, /*enable=*/1,
                        /*etype=*/ompt_callback_target_submit);
  }

  return ompt_set_always;
}

/////// HOST-RELATED //////

static void on_ompt_callback_thread_begin(ompt_thread_t thread_type,
                                          ompt_data_t *thread_data) {
  OmptCallbackHandler::get().handleThreadBegin(thread_type, thread_data);
}

static void on_ompt_callback_thread_end(ompt_data_t *thread_data) {
  OmptCallbackHandler::get().handleThreadEnd(thread_data);
}

static void on_ompt_callback_parallel_begin(
    ompt_data_t *encountering_task_data,
    const ompt_frame_t *encountering_task_frame, ompt_data_t *parallel_data,
    unsigned int requested_parallelism, int flags, const void *codeptr_ra) {
  OmptCallbackHandler::get().handleParallelBegin(
      encountering_task_data, encountering_task_frame, parallel_data,
      requested_parallelism, flags, codeptr_ra);
}

static void on_ompt_callback_parallel_end(ompt_data_t *parallel_data,
                                          ompt_data_t *encountering_task_data,
                                          int flags, const void *codeptr_ra) {
  OmptCallbackHandler::get().handleParallelEnd(
      parallel_data, encountering_task_data, flags, codeptr_ra);
}

static void
on_ompt_callback_task_create(ompt_data_t *encountering_task_data,
                             const ompt_frame_t *encountering_task_frame,
                             ompt_data_t *new_task_data, int flags,
                             int has_dependences, const void *codeptr_ra) {
  OmptCallbackHandler::get().handleTaskCreate(
      encountering_task_data, encountering_task_frame, new_task_data, flags,
      has_dependences, codeptr_ra);
}

static void on_ompt_callback_task_schedule(ompt_data_t *prior_task_data,
                                           ompt_task_status_t prior_task_status,
                                           ompt_data_t *next_task_data) {
  OmptCallbackHandler::get().handleTaskSchedule(
      prior_task_data, prior_task_status, next_task_data);
}

static void on_ompt_callback_implicit_task(ompt_scope_endpoint_t endpoint,
                                           ompt_data_t *parallel_data,
                                           ompt_data_t *task_data,
                                           unsigned int actual_parallelism,
                                           unsigned int index, int flags) {
  OmptCallbackHandler::get().handleImplicitTask(
      endpoint, parallel_data, task_data, actual_parallelism, index, flags);
}

// Callbacks as of Table 19.4, which are not considered required for a minimal
// conforming OMPT implementation.
static void on_ompt_callback_work(ompt_work_t work_type,
                                  ompt_scope_endpoint_t endpoint,
                                  ompt_data_t *parallel_data,
                                  ompt_data_t *task_data, uint64_t count,
                                  const void *codeptr_ra) {
  OmptCallbackHandler::get().handleWork(work_type, endpoint, parallel_data,
                                        task_data, count, codeptr_ra);
}

static void on_ompt_callback_dispatch(ompt_data_t *parallel_data,
                                      ompt_data_t *task_data,
                                      ompt_dispatch_t kind,
                                      ompt_data_t instance) {
  OmptCallbackHandler::get().handleDispatch(parallel_data, task_data, kind,
                                            instance);
}

static void on_ompt_callback_sync_region(ompt_sync_region_t kind,
                                         ompt_scope_endpoint_t endpoint,
                                         ompt_data_t *parallel_data,
                                         ompt_data_t *task_data,
                                         const void *codeptr_ra) {
  OmptCallbackHandler::get().handleSyncRegion(kind, endpoint, parallel_data,
                                              task_data, codeptr_ra);
}

/////// DEVICE-RELATED //////

// Synchronous callbacks
static void on_ompt_callback_device_initialize(int device_num, const char *type,
                                               ompt_device_t *device,
                                               ompt_function_lookup_t lookup,
                                               const char *documentation) {
  OmptCallbackHandler::get().handleDeviceInitialize(device_num, type, device,
                                                    lookup, documentation);
  if (!UseTracing)
    return;

  if (!lookup) {
    printf("Trace collection disabled on device %d\n", device_num);
    return;
  }

  ompt_set_trace_ompt = (ompt_set_trace_ompt_t)lookup("ompt_set_trace_ompt");
  ompt_start_trace = (ompt_start_trace_t)lookup("ompt_start_trace");
  ompt_flush_trace = (ompt_flush_trace_t)lookup("ompt_flush_trace");
  ompt_stop_trace = (ompt_stop_trace_t)lookup("ompt_stop_trace");
  ompt_get_record_ompt = (ompt_get_record_ompt_t)lookup("ompt_get_record_ompt");
  ompt_advance_buffer_cursor =
      (ompt_advance_buffer_cursor_t)lookup("ompt_advance_buffer_cursor");

  ompt_get_record_type_fn =
      (ompt_get_record_type_t)lookup("ompt_get_record_type");
  if (!ompt_get_record_type_fn) {
    printf("Warning: No function ompt_get_record_type found in device "
           "callbacks\n");
  }

  static bool IsDeviceMapInitialized = false;
  if (!IsDeviceMapInitialized) {
    TracedDevices = std::make_unique<OmptDeviceSetTy>();
    IsDeviceMapInitialized = true;
  }

  set_trace_ompt(device);

  // In many scenarios, this is a good place to start the
  // trace. If start_trace is called from the main program before this
  // callback is dispatched, the start_trace handle will be null. This
  // is because this device_init callback is invoked during the first
  // target construct implementation.

  start_trace(device);
}

static void on_ompt_callback_device_finalize(int device_num) {
  OmptCallbackHandler::get().handleDeviceFinalize(device_num);
}

static void on_ompt_callback_device_load(int device_num, const char *filename,
                                         int64_t offset_in_file,
                                         void *vma_in_file, size_t bytes,
                                         void *host_addr, void *device_addr,
                                         uint64_t module_id) {
  OmptCallbackHandler::get().handleDeviceLoad(
      device_num, filename, offset_in_file, vma_in_file, bytes, host_addr,
      device_addr, module_id);
}

static void on_ompt_callback_device_unload(int device_num, uint64_t module_id) {
  OmptCallbackHandler::get().handleDeviceUnload(device_num, module_id);
}

static void on_ompt_callback_target_data_op(
    ompt_id_t target_id, ompt_id_t host_op_id, ompt_target_data_op_t optype,
    void *src_addr, int src_device_num, void *dest_addr, int dest_device_num,
    size_t bytes, const void *codeptr_ra) {
  OmptCallbackHandler::get().handleTargetDataOp(
      target_id, host_op_id, optype, src_addr, src_device_num, dest_addr,
      dest_device_num, bytes, codeptr_ra);
}

static void on_ompt_callback_target(ompt_target_t kind,
                                    ompt_scope_endpoint_t endpoint,
                                    int device_num, ompt_data_t *task_data,
                                    ompt_id_t target_id,
                                    const void *codeptr_ra) {
  OmptCallbackHandler::get().handleTarget(kind, endpoint, device_num, task_data,
                                          target_id, codeptr_ra);
}

static void on_ompt_callback_target_submit(ompt_id_t target_id,
                                           ompt_id_t host_op_id,
                                           unsigned int requested_num_teams) {
  OmptCallbackHandler::get().handleTargetSubmit(target_id, host_op_id,
                                                requested_num_teams);
}

static void on_ompt_callback_target_data_op_emi(
    ompt_scope_endpoint_t endpoint, ompt_data_t *target_task_data,
    ompt_data_t *target_data, ompt_id_t *host_op_id,
    ompt_target_data_op_t optype, void *src_addr, int src_device_num,
    void *dest_addr, int dest_device_num, size_t bytes,
    const void *codeptr_ra) {
  assert(codeptr_ra != 0 && "Unexpected null codeptr");
  // Both src and dest must not be null
  // However, for omp_target_alloc only the END call holds a value for one of
  // the two entries
  if (optype != ompt_target_data_alloc)
    assert((src_addr != 0 || dest_addr != 0) && "Both src and dest addr null");
  if (endpoint == ompt_scope_begin)
    *host_op_id = NextOpId.fetch_add(1, std::memory_order_relaxed);
  OmptCallbackHandler::get().handleTargetDataOpEmi(
      endpoint, target_task_data, target_data, host_op_id, optype, src_addr,
      src_device_num, dest_addr, dest_device_num, bytes, codeptr_ra);
}

static void on_ompt_callback_target_emi(ompt_target_t kind,
                                        ompt_scope_endpoint_t endpoint,
                                        int device_num, ompt_data_t *task_data,
                                        ompt_data_t *target_task_data,
                                        ompt_data_t *target_data,
                                        const void *codeptr_ra) {
  assert(codeptr_ra != 0 && "Unexpected null codeptr");
  if (endpoint == ompt_scope_begin)
    target_data->value = NextOpId.fetch_add(1, std::memory_order_relaxed);
  OmptCallbackHandler::get().handleTargetEmi(kind, endpoint, device_num,
                                             task_data, target_task_data,
                                             target_data, codeptr_ra);
}

static void on_ompt_callback_target_submit_emi(
    ompt_scope_endpoint_t endpoint, ompt_data_t *target_data,
    ompt_id_t *host_op_id, unsigned int requested_num_teams) {
  OmptCallbackHandler::get().handleTargetSubmitEmi(
      endpoint, target_data, host_op_id, requested_num_teams);
}

static void on_ompt_callback_target_map(ompt_id_t target_id,
                                        unsigned int nitems, void **host_addr,
                                        void **device_addr, size_t *bytes,
                                        unsigned int *mapping_flags,
                                        const void *codeptr_ra) {
  assert(0 && "Target map callback is unimplemented");
}

static void on_ompt_callback_target_map_emi(ompt_data_t *target_data,
                                            unsigned int nitems,
                                            void **host_addr,
                                            void **device_addr, size_t *bytes,
                                            unsigned int *mapping_flags,
                                            const void *codeptr_ra) {
  assert(0 && "Target map emi callback is unimplemented");
}

/// Load the value of a given boolean environmental variable.
bool getBoolEnvironmentVariable(const char *VariableName) {
  if (VariableName == nullptr)
    return false;
  if (const char *EnvValue = std::getenv(VariableName)) {
    std::string S{EnvValue};
    for (auto &C : S)
      C = (char)std::tolower(C);
    if (S == "1" || S == "on" || S == "true" || S == "yes")
      return true;
  }
  return false;
}

/// Called by the OMP runtime to initialize the OMPT
int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_set_callback_t ompt_set_callback = nullptr;
  ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
  if (!ompt_set_callback)
    return 0; // failure

  UseEMICallbacks = getBoolEnvironmentVariable("OMPTEST_USE_OMPT_EMI");
  UseTracing = getBoolEnvironmentVariable("OMPTEST_USE_OMPT_TRACING");
  RunAsTestSuite = getBoolEnvironmentVariable("OMPTEST_RUN_AS_TESTSUITE");
  ColoredLog = getBoolEnvironmentVariable("OMPTEST_LOG_COLORED");

  register_ompt_callback(ompt_callback_thread_begin);
  register_ompt_callback(ompt_callback_thread_end);
  register_ompt_callback(ompt_callback_parallel_begin);
  register_ompt_callback(ompt_callback_parallel_end);
  register_ompt_callback(ompt_callback_work);
  register_ompt_callback(ompt_callback_dispatch);
  register_ompt_callback(ompt_callback_task_create);
  // register_ompt_callback(ompt_callback_dependences);
  // register_ompt_callback(ompt_callback_task_dependence);
  register_ompt_callback(ompt_callback_task_schedule);
  register_ompt_callback(ompt_callback_implicit_task);
  // register_ompt_callback(ompt_callback_masked);
  register_ompt_callback(ompt_callback_sync_region);
  // register_ompt_callback(ompt_callback_mutex_acquire);
  // register_ompt_callback(ompt_callback_mutex);
  // register_ompt_callback(ompt_callback_nestLock);
  // register_ompt_callback(ompt_callback_flush);
  // register_ompt_callback(ompt_callback_cancel);
  register_ompt_callback(ompt_callback_device_initialize);
  register_ompt_callback(ompt_callback_device_finalize);
  register_ompt_callback(ompt_callback_device_load);
  register_ompt_callback(ompt_callback_device_unload);

  if (UseEMICallbacks) {
    register_ompt_callback(ompt_callback_target_emi);
    register_ompt_callback(ompt_callback_target_submit_emi);
    register_ompt_callback(ompt_callback_target_data_op_emi);
    register_ompt_callback(ompt_callback_target_map_emi);
  } else {
    register_ompt_callback(ompt_callback_target);
    register_ompt_callback(ompt_callback_target_submit);
    register_ompt_callback(ompt_callback_target_data_op);
    register_ompt_callback(ompt_callback_target_map);
  }

  // Construct & subscribe the reporter, so it gets notified of events
  EventReporter = new OmptEventReporter();
  OmptCallbackHandler::get().subscribe(EventReporter);

  if (RunAsTestSuite)
    EventReporter->setActive(false);

  return 1; // success
}

void ompt_finalize(ompt_data_t *tool_data) {
  assert(Handler && "Callback handler should be present at this point");
  assert(EventReporter && "EventReporter should be present at this point");
  delete Handler;
  delete EventReporter;
}

#ifdef __cplusplus
extern "C" {
#endif
/// Called from the OMP Runtime to start / initialize the tool
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t ompt_start_tool_result = {
      &ompt_initialize, &ompt_finalize, {0}};
  return &ompt_start_tool_result;
}

int start_trace(ompt_device_t *Device) {
  if (!ompt_start_trace)
    return 0;

  // Start tracing this device (add to set)
  assert(TracedDevices->find(Device) == TracedDevices->end() &&
         "Device already present in the map");
  TracedDevices->insert(Device);

  return ompt_start_trace(Device, &on_ompt_callback_buffer_request,
                          &on_ompt_callback_buffer_complete);
}

int flush_trace(ompt_device_t *Device) {
  if (!ompt_flush_trace)
    return 0;
  return ompt_flush_trace(Device);
}

int flush_traced_devices() {
  if (!ompt_flush_trace || TracedDevices == nullptr)
    return 0;

  size_t NumFlushedDevices = 0;
  for (auto Device : *TracedDevices)
    if (ompt_flush_trace(Device) == 1)
      ++NumFlushedDevices;

  // Provide time to process triggered assert events
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  return (NumFlushedDevices == TracedDevices->size());
}

int stop_trace(ompt_device_t *Device) {
  if (!ompt_stop_trace)
    return 0;

  // Stop tracing this device (erase from set)
  assert(TracedDevices->find(Device) != TracedDevices->end() &&
         "Device not present in the map");
  TracedDevices->erase(Device);

  return ompt_stop_trace(Device);
}

// This is primarily used to stop unwanted prints from happening.
void libomptest_global_eventreporter_set_active(bool State) {
  assert(EventReporter && "EventReporter should be present at this point");
  EventReporter->setActive(State);
}
#ifdef __cplusplus
}
#endif
