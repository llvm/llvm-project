//===-- OmptCallback.cpp - Target independent OpenMP target RTL --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of OMPT callback interfaces for target independent layer
//
//===----------------------------------------------------------------------===//

#ifdef OMPT_SUPPORT

#include <assert.h>
#include <atomic>
#include <cstring>
#include <mutex>

//****************************************************************************
// local include files
//****************************************************************************

#include <omp-tools.h>

#include "Debug.h"
#include "ompt-connector.h"
#include "ompt_buffer_mgr.h"
#include "ompt_callback.h"
#include "ompt_device_callbacks.h"
#include "private.h"

/*******************************************************************************
 * macros
 ******************************************************************************/

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

#define OMPT_CALLBACK_AVAILABLE(fn) (OmptEnabled && fn)
#define OMPT_CALLBACK(fn, args) OmptDeviceCallbacks.fn args
#define fnptr_to_ptr(x) ((void *)(uint64_t)x)

/*******************************************************************************
 * type declarations
 ******************************************************************************/

/// Used to maintain the finalization function that is received
/// from the plugin during connect
class LibomptargetRtlFinalizer {
public:
  LibomptargetRtlFinalizer() : RtlFinalization(nullptr) {}
  void registerRtl(ompt_finalize_t FinalizationFunction) {
    assert((RtlFinalization == nullptr) &&
           "RTL finalization may only be registered once");
    RtlFinalization = FinalizationFunction;
  }
  void finalize() {
    if (RtlFinalization)
      RtlFinalization(nullptr /* tool_data */);
    RtlFinalization = nullptr;
  }

private:
  ompt_finalize_t RtlFinalization;
};

/// Object that will maintain the RTL finalizer from the plugin
static LibomptargetRtlFinalizer LibraryFinalizer;

typedef int (*ompt_set_frame_enter_t)(void *addr, int flags, int state);

typedef ompt_data_t *(*ompt_get_task_data_t)();
typedef ompt_data_t *(*ompt_get_target_task_data_t)();

/*****************************************************************************
 * global data
 *****************************************************************************/

/// Used to indicate whether OMPT was enabled for this library
bool OmptEnabled = false;

/// Object maintaining all the callbacks for this library
OmptDeviceCallbacksTy OmptDeviceCallbacks;

OmptTracingBufferMgr ompt_trace_record_buffer_mgr;

/*****************************************************************************
 * private data
 *****************************************************************************/

static ompt_set_frame_enter_t ompt_set_frame_enter_fn = 0;
static ompt_get_task_data_t ompt_get_task_data_fn = 0;
static ompt_get_target_task_data_t ompt_get_target_task_data_fn = 0;

static std::atomic<uint64_t> unique_id_ticket(1);

// Mutexes to serialize entry points invocation
static std::mutex set_trace_mutex;
// Serialize start/stop/flush
static std::mutex start_stop_flush_trace_mutex;

/*****************************************************************************
 * Thread local data
 *****************************************************************************/

thread_local OmptInterface ompt_interface;

static thread_local uint64_t ompt_target_region_opid = 1;
static thread_local ompt_data_t ompt_target_data = ompt_data_none;
static thread_local ompt_data_t *ompt_task_data = 0;
static thread_local ompt_data_t *ompt_target_task_data = 0;
static thread_local ompt_id_t host_op_id = 0;

// Thread local variables used by the plugin to communicate OMPT information
// that are then used to populate trace records. This method assumes a
// synchronous implementation, otherwise it won't work.
static thread_local uint32_t ompt_num_granted_teams = 0;
static thread_local uint64_t ompt_tr_start_time = 0;
static thread_local uint64_t ompt_tr_end_time = 0;

/*****************************************************************************
 * OMPT callbacks
 *****************************************************************************/

void OmptInterface::ompt_state_set_helper(void *enter_frame, void *codeptr_ra,
                                          int flags, int state) {
  _enter_frame = enter_frame;
  _codeptr_ra = codeptr_ra;
  if (ompt_set_frame_enter_fn) {
    _state = ompt_set_frame_enter_fn(_enter_frame, flags, state);
  }

  return;
}

void OmptInterface::ompt_state_set(void *enter_frame, void *codeptr_ra) {
  ompt_state_set_helper(enter_frame, codeptr_ra, OMPT_FRAME_FLAGS,
                        ompt_state_work_parallel);
}

void OmptInterface::ompt_state_clear(void) {
  ompt_state_set_helper(0, 0, 0, _state);
}

/// This is the function called by the higher layer (libomp) responsible
/// for initializing OMPT in this library. This is passed to libomp
/// as part of the OMPT connector object.
/// \p lookup to be used to query callbacks registered with libomp
/// \p initial_device_num Initial device num provided by libomp
/// \p tool_data as provided by the tool
static int ompt_libomptarget_initialize(ompt_function_lookup_t lookup,
                                        int initial_device_num,
                                        ompt_data_t *tool_data) {
  DP("enter ompt_libomptarget_initialize!\n");
  OmptEnabled = true;

#define ompt_bind_name(fn)                                                     \
  fn##_fn = (fn##_t)lookup(#fn);                                               \
  DP("%s=%p\n", #fn, fnptr_to_ptr(fn##_fn));

  ompt_bind_name(ompt_set_frame_enter);
  ompt_bind_name(ompt_get_task_data);
  ompt_bind_name(ompt_get_target_task_data);

#undef ompt_bind_name

  // The lookup parameter is provided by libomp which already has the
  // tool callbacks registered at this point. The registration call
  // below causes the same callback functions to be registered in
  // libomptarget as well
  OmptDeviceCallbacks.registerCallbacks(lookup);

  DP("exit ompt_libomptarget_initialize!\n");
  return 0;
}

/// This function is passed to libomp as part of the OMPT connector object.
/// It is called by libomp during finalization of OMPT in libomptarget.
static void ompt_libomptarget_finalize(ompt_data_t *data) {
  DP("enter ompt_libomptarget_finalize!\n");
  // Before disabling OMPT, call the finalizer (of the plugin) that was
  // registered with this library
  LibraryFinalizer.finalize();
  OmptEnabled = false;
  DP("exit ompt_libomptarget_finalize!\n");
}

/*****************************************************************************
 * OMPT private operations
 *****************************************************************************/
/// Used to initialize callbacks implemented by the tool. This interface
/// will lookup the callbacks table in libomp and assign them to the callbacks
/// maintained in libomptarget.
void InitOmptLibomp() {
  DP("OMPT: Enter InitOmptLibomp\n");
  // Connect with libomp
  static OmptLibraryConnectorTy LibompConnector("libomp");
  static ompt_start_tool_result_t OmptResult;

  // Initialize OmptResult with the init and fini functions that will be
  // called by the connector
  OmptResult.initialize = ompt_libomptarget_initialize;
  OmptResult.finalize = ompt_libomptarget_finalize;
  OmptResult.tool_data.value = 0;

  // Initialize the device callbacks first
  OmptDeviceCallbacks.init();

  // Now call connect that causes the above init/fini functions to be called
  LibompConnector.connect(&OmptResult);
  DP("OMPT: Exit InitOmptLibomp\n");
}

static uint64_t id_create() { return unique_id_ticket.fetch_add(1); }

static uint64_t opid_create() {
  host_op_id = id_create();
  return host_op_id;
}

static uint64_t opid_get() { return host_op_id; }

static uint64_t regionid_create() {
  ompt_target_data.value = id_create();
  return ompt_target_data.value;
}

static uint64_t regionid_get() { return ompt_target_data.value; }

void OmptInterface::target_region_begin() {
  // set up task region state
  ompt_task_data = ompt_get_task_data_fn();
  ompt_target_task_data = ompt_get_target_task_data_fn();

  ompt_target_data = ompt_data_none;
}

void OmptInterface::target_region_announce(const char *name) {
  DP("in OmptInterface::target_region_%s target_id=%lu\n", name,
     ompt_target_data.value);
}

void OmptInterface::target_region_end() {
  ompt_task_data = 0;
  ompt_target_task_data = 0;
  ompt_target_data = ompt_data_none;
}

void OmptInterface::target_operation_begin() {
  DP("in ompt_target_region_begin (ompt_target_region_opid = %lu)\n",
     ompt_target_data.value);
}

void OmptInterface::target_operation_end() {
  DP("in ompt_target_region_end (ompt_target_region_opid = %lu)\n",
     ompt_target_data.value);
}

/*****************************************************************************
 * OMPT public operations
 *****************************************************************************/

// FIXME: optional implementation of target map?

void OmptInterface::target_data_alloc_begin(int64_t device_id,
                                            void *hst_ptr_begin, size_t size,
                                            void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_data_op_emi(
      ompt_scope_begin, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_alloc, /*src_addr=*/hst_ptr_begin,
      /*src_device_num=*/omp_get_initial_device(), /*dest_addr=*/nullptr,
      /*dest_device_num=*/device_id, size, codeptr, opid_create,
      &ompt_target_region_opid);
  target_operation_begin();
}

void OmptInterface::target_data_alloc_end(int64_t device_id,
                                          void *hst_ptr_begin,
                                          void *tgt_ptr_begin, size_t size,
                                          void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_data_op_emi(
      ompt_scope_end, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_alloc, /*src_addr=*/hst_ptr_begin,
      /*src_device_num=*/omp_get_initial_device(),
      /*dest_addr=*/tgt_ptr_begin, /*dest_device_num=*/device_id, size, codeptr,
      opid_get, &ompt_target_region_opid);
  target_operation_end();
}

void OmptInterface::target_data_submit_begin(int64_t device_id,
                                             void *hst_ptr_begin,
                                             void *tgt_ptr_begin, size_t size,
                                             void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_data_op_emi(
      ompt_scope_begin, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_transfer_to_device, /*src_addr=*/hst_ptr_begin,
      /*src_device_num=*/omp_get_initial_device(),
      /*dest_addr=*/tgt_ptr_begin, /*dest_device_num=*/device_id, size, codeptr,
      opid_create, &ompt_target_region_opid);
  target_operation_begin();
}

void OmptInterface::target_data_submit_end(int64_t device_id,
                                           void *hst_ptr_begin,
                                           void *tgt_ptr_begin, size_t size,
                                           void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_data_op_emi(
      ompt_scope_end, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_transfer_to_device, /*src_addr=*/hst_ptr_begin,
      /*src_device_num=*/omp_get_initial_device(),
      /*dest_addr=*/tgt_ptr_begin, /*dest_device_num=*/device_id, size, codeptr,
      opid_get, &ompt_target_region_opid);
  target_operation_end();
}

void OmptInterface::target_data_delete_begin(int64_t device_id,
                                             void *tgt_ptr_begin,
                                             void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_data_op_emi(
      ompt_scope_begin, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_delete, /*src_addr=*/tgt_ptr_begin,
      /*src_device_num=*/device_id, /*dest_addr=*/nullptr,
      /*dest_device_num=*/-1, /*size=*/0, codeptr, opid_create,
      &ompt_target_region_opid);
  target_operation_begin();
}

void OmptInterface::target_data_delete_end(int64_t device_id,
                                           void *tgt_ptr_begin, void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_data_op_emi(
      ompt_scope_end, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_delete, /*src_addr=*/tgt_ptr_begin,
      /*src_device_num=*/device_id, /*dest_addr=*/nullptr,
      /*dest_device_num=*/-1, /*size=*/0, codeptr, opid_get,
      &ompt_target_region_opid);
  target_operation_end();
}

void OmptInterface::target_data_retrieve_begin(int64_t device_id,
                                               void *hst_ptr_begin,
                                               void *tgt_ptr_begin, size_t size,
                                               void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_data_op_emi(
      ompt_scope_begin, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_transfer_from_device, /*src_addr=*/tgt_ptr_begin,
      /*src_device_num=*/device_id, /*dest_addr=*/hst_ptr_begin,
      /*dest_device_num=*/omp_get_initial_device(), size, codeptr, opid_create,
      &ompt_target_region_opid);
  target_operation_begin();
}

void OmptInterface::target_data_retrieve_end(int64_t device_id,
                                             void *hst_ptr_begin,
                                             void *tgt_ptr_begin, size_t size,
                                             void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_data_op_emi(
      ompt_scope_end, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_transfer_from_device, /*src_addr=*/tgt_ptr_begin,
      /*src_device_num=*/device_id, /*dest_addr=*/hst_ptr_begin,
      /*dest_device_num=*/omp_get_initial_device(), size, codeptr, opid_get,
      &ompt_target_region_opid);
  target_operation_end();
}

ompt_record_ompt_t *OmptInterface::target_data_submit_trace_record_gen(
    ompt_target_data_op_t data_op, void *src_addr, int64_t src_device_num,
    void *dest_addr, int64_t dest_device_num, size_t bytes) {
  if (!OmptDeviceCallbacks.is_tracing_enabled() ||
      (!OmptDeviceCallbacks.is_tracing_type_enabled(
           ompt_callback_target_data_op) &&
       !OmptDeviceCallbacks.is_tracing_type_enabled(
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

void OmptInterface::set_trace_record_target_data_op(
    ompt_record_target_data_op_t *rec, ompt_target_data_op_t data_op,
    void *src_addr, int64_t src_device_num, void *dest_addr,
    int64_t dest_device_num, size_t bytes) {
  rec->host_op_id = ompt_target_region_opid;
  rec->optype = data_op;
  rec->src_addr = src_addr;
  rec->src_device_num = src_device_num;
  rec->dest_addr = dest_addr;
  rec->dest_device_num = dest_device_num;
  rec->bytes = bytes;
  rec->end_time = ompt_tr_end_time;
  rec->codeptr_ra = _codeptr_ra;
}

void OmptInterface::target_submit_begin(unsigned int num_teams) {
  OmptDeviceCallbacks.ompt_callback_target_submit_emi(
      ompt_scope_begin, &ompt_target_data, num_teams, opid_create,
      &ompt_target_region_opid);
}

void OmptInterface::target_submit_end(unsigned int num_teams) {
  OmptDeviceCallbacks.ompt_callback_target_submit_emi(
      ompt_scope_end, &ompt_target_data, num_teams, opid_get,
      &ompt_target_region_opid);
}

ompt_record_ompt_t *
OmptInterface::target_submit_trace_record_gen(unsigned int num_teams) {
  if (!OmptDeviceCallbacks.is_tracing_enabled() ||
      (!OmptDeviceCallbacks.is_tracing_type_enabled(
           ompt_callback_target_submit) &&
       !OmptDeviceCallbacks.is_tracing_type_enabled(
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

void OmptInterface::set_trace_record_target_kernel(
    ompt_record_target_kernel_t *rec, unsigned int num_teams) {
  rec->host_op_id = ompt_target_region_opid;
  rec->requested_num_teams = num_teams;
  rec->granted_num_teams = ompt_num_granted_teams;
  rec->end_time = ompt_tr_end_time;
}

void OmptInterface::target_data_enter_begin(int64_t device_id, void *codeptr) {
  target_region_begin();
  OmptDeviceCallbacks.ompt_callback_target_emi(
      ompt_target_enter_data, ompt_scope_begin, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_create);
}

void OmptInterface::target_data_enter_end(int64_t device_id, void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_emi(
      ompt_target_enter_data, ompt_scope_end, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_get);
  target_region_end();
}

void OmptInterface::target_data_exit_begin(int64_t device_id, void *codeptr) {
  target_region_begin();
  OmptDeviceCallbacks.ompt_callback_target_emi(
      ompt_target_exit_data, ompt_scope_begin, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_create);
  target_region_announce("begin");
}

void OmptInterface::target_data_exit_end(int64_t device_id, void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_emi(
      ompt_target_exit_data, ompt_scope_end, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_get);
  target_region_end();
}

void OmptInterface::target_update_begin(int64_t device_id, void *codeptr) {
  target_region_begin();
  OmptDeviceCallbacks.ompt_callback_target_emi(
      ompt_target_update, ompt_scope_begin, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_create);
  target_region_announce("begin");
}

void OmptInterface::target_update_end(int64_t device_id, void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_emi(
      ompt_target_update, ompt_scope_end, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_get);
  target_region_end();
}

void OmptInterface::target_begin(int64_t device_id, void *codeptr) {
  target_region_begin();
  OmptDeviceCallbacks.ompt_callback_target_emi(
      ompt_target, ompt_scope_begin, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_create);
  target_region_announce("begin");
}

void OmptInterface::target_end(int64_t device_id, void *codeptr) {
  OmptDeviceCallbacks.ompt_callback_target_emi(
      ompt_target, ompt_scope_end, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_get);
  target_region_end();
}

ompt_record_ompt_t *
OmptInterface::target_trace_record_gen(int64_t device_id, ompt_target_t kind,
                                       ompt_scope_endpoint_t endpoint,
                                       void *code) {
  if (!OmptDeviceCallbacks.is_tracing_enabled() ||
      (!OmptDeviceCallbacks.is_tracing_type_enabled(ompt_callback_target) &&
       !OmptDeviceCallbacks.is_tracing_type_enabled(ompt_callback_target_emi)))
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

void OmptInterface::set_trace_record_target(ompt_record_target_t *rec,
                                            int64_t device_id,
                                            ompt_target_t kind,
                                            ompt_scope_endpoint_t endpoint,
                                            void *code) {
  rec->kind = kind;
  rec->endpoint = endpoint;
  rec->device_num = device_id;
  assert(ompt_task_data);
  rec->task_id = ompt_task_data->value;
  rec->target_id = ompt_target_data.value;
  rec->codeptr_ra = code;
}

void OmptInterface::set_trace_record_common(ompt_record_ompt_t *data_ptr,
                                            ompt_callbacks_t cbt) {
  data_ptr->type = cbt;
  if (cbt == ompt_callback_target)
    data_ptr->time = 0; // Currently, no consumer, so no need to set it
  else
    data_ptr->time = ompt_tr_start_time;
  data_ptr->thread_id = 0; // TODO
  data_ptr->target_id = ompt_target_data.value;
}

/*****************************************************************************
 * OMPT interface operations
 *****************************************************************************/

static void LIBOMPTARGET_GET_TARGET_OPID(uint64_t *device_num,
                                         ompt_id_t *target_id,
                                         ompt_id_t *host_op_id) {
  *host_op_id = ompt_target_region_opid;
}

// Today, this is not called from libomptarget
ompt_device *OmptDeviceCallbacksTy::lookup_device(int device_num) {
  assert(0 && "Lookup device should be invoked in the plugin");
  return nullptr;
}

ompt_interface_fn_t
OmptDeviceCallbacksTy::doLookup(const char *InterfaceFunctionName) {
  if (strcmp(InterfaceFunctionName, stringify(LIBOMPTARGET_GET_TARGET_OPID)) ==
      0)
    return (ompt_interface_fn_t)LIBOMPTARGET_GET_TARGET_OPID;

  return OmptDeviceCallbacks.lookupCallback(InterfaceFunctionName);
}

extern "C" {
// Device-independent entry point for ompt_set_trace_ompt
ompt_set_result_t libomptarget_ompt_set_trace_ompt(ompt_device_t *device,
                                                   unsigned int enable,
                                                   unsigned int etype) {
  std::unique_lock<std::mutex> lck(set_trace_mutex);
  return OmptDeviceCallbacks.set_trace_ompt(device, enable, etype);
}

// Device-independent entry point for ompt_start_trace
int libomptarget_ompt_start_trace(ompt_callback_buffer_request_t request,
                                  ompt_callback_buffer_complete_t complete) {
  std::unique_lock<std::mutex> lck(start_stop_flush_trace_mutex);
  OmptDeviceCallbacks.set_buffer_request(request);
  OmptDeviceCallbacks.set_buffer_complete(complete);
  if (request && complete) {
    OmptDeviceCallbacks.set_tracing_enabled(true);
    ompt_trace_record_buffer_mgr.startHelperThreads();
    return 1; // success
  }
  return 0; // failure
}

// Device-independent entry point for ompt_flush_trace
int libomptarget_ompt_flush_trace(ompt_device_t *device) {
  std::unique_lock<std::mutex> lck(start_stop_flush_trace_mutex);
  return ompt_trace_record_buffer_mgr.flushAllBuffers(device);
}

// Device independent entry point for ompt_stop_trace
int libomptarget_ompt_stop_trace(ompt_device_t *device) {
  std::unique_lock<std::mutex> lck(start_stop_flush_trace_mutex);
  int status = ompt_trace_record_buffer_mgr.flushAllBuffers(device);
  // TODO shutdown should perhaps return a status
  ompt_trace_record_buffer_mgr.shutdownHelperThreads();
  OmptDeviceCallbacks.set_tracing_enabled(false);
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

  size_t sz = sizeof(ompt_record_ompt_t);
  *next = (ompt_buffer_cursor_t)(curr_rec + sz);
  DP("Advanced buffer pointer by %lu bytes to %p\n", sz, curr_rec + sz);
  return true;
}

// This function is invoked before the kernel launch. So when the
// trace record is populated after kernel completion,
// ompt_num_granted_teams is already updated
void libomptarget_ompt_set_granted_teams(uint32_t num_teams) {
  ompt_num_granted_teams = num_teams;
}

// Assume a synchronous implementation and set thread local variables to track
// timestamps. The thread local variables can then be used to populate trace
// records.
void libomptarget_ompt_set_timestamp(uint64_t start, uint64_t end) {
  ompt_tr_start_time = start;
  ompt_tr_end_time = end;
}

// Device-independent entry point to query for the trace format used.
// Currently, only OMPT format is supported.
ompt_record_t libomptarget_ompt_get_record_type(ompt_buffer_t *buffer,
                                                ompt_buffer_cursor_t current) {
  // TODO: When different OMPT trace buffer formats supported, this needs to be
  // fixed.
  return ompt_record_t::ompt_record_ompt;
}

/// Used for connecting libomptarget with a plugin
void ompt_libomptarget_connect(ompt_start_tool_result_t *result) {
  DP("OMPT: Enter ompt_libomptarget_connect\n");
  if (OmptEnabled && result) {
    // Cache the fini function so that it can be invoked on exit
    LibraryFinalizer.registerRtl(result->finalize);
    // Invoke the provided init function with the lookup function maintained
    // in this library so that callbacks maintained by this library are
    // retrieved.
    result->initialize(OmptDeviceCallbacksTy::doLookup,
                       0 /* initial_device_num */, nullptr /* tool_data */);
  }
  DP("OMPT: Leave ompt_libomptarget_connect\n");
}
}

#pragma pop_macro("DEBUG_PREFIX")

#else
extern "C" {
/// Dummy definition when OMPT is disabled
void ompt_libomptarget_connect() {}
}
#endif // OMPT_SUPPORT
