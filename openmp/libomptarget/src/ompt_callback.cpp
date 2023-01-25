//===-- ompt_callback.cpp - Target independent OpenMP target RTL -- C++ -*-===//
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

#include <assert.h>
#include <atomic>
#include <cstring>
#include <mutex>

//****************************************************************************
// local include files
//****************************************************************************

#include <omp-tools.h>

#include "ompt_callback.h"
#include "private.h"

#include <ompt-connector.h>
#include <ompt_buffer_mgr.h>
#include <ompt_device_callbacks.h>

/*******************************************************************************
 * macros
 *******************************************************************************/

#define OMPT_CALLBACK_AVAILABLE(fn) (ompt_enabled && fn)
#define OMPT_CALLBACK(fn, args) ompt_device_callbacks.fn args
#define fnptr_to_ptr(x) ((void *)(uint64_t)x)

/*******************************************************************************
 * type declarations
 *******************************************************************************/

class libomptarget_rtl_finalizer_t {
public:
  libomptarget_rtl_finalizer_t() : fn(0){};

  void register_rtl(ompt_finalize_t _fn) {
    assert(fn == 0);
    fn = _fn;
  };

  void finalize() {
    if (fn)
      fn(NULL);
    fn = 0;
  };

  ompt_finalize_t fn;
};

typedef int (*ompt_set_frame_enter_t)(void *addr, int flags, int state);

typedef ompt_data_t *(*ompt_get_task_data_t)();
typedef ompt_data_t *(*ompt_get_target_task_data_t)();

/*****************************************************************************
 * global data
 *****************************************************************************/

bool ompt_enabled = false;

ompt_device_callbacks_t ompt_device_callbacks;

OmptTracingBufferMgr ompt_trace_record_buffer_mgr;

/*****************************************************************************
 * private data
 *****************************************************************************/

static ompt_set_frame_enter_t ompt_set_frame_enter_fn = 0;
static ompt_get_task_data_t ompt_get_task_data_fn = 0;
static ompt_get_target_task_data_t ompt_get_target_task_data_fn = 0;

static libomptarget_rtl_finalizer_t libomptarget_rtl_finalizer;

const char *ompt_device_callbacks_t::documentation = 0;

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

/*****************************************************************************
 * OMPT private operations
 *****************************************************************************/

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
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
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
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
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
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
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
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
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
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
      ompt_scope_begin, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_delete, /*src_addr=*/tgt_ptr_begin,
      /*src_device_num=*/device_id, /*dest_addr=*/nullptr,
      /*dest_device_num=*/-1, /*size=*/0, codeptr, opid_create,
      &ompt_target_region_opid);
  target_operation_begin();
}

void OmptInterface::target_data_delete_end(int64_t device_id,
                                           void *tgt_ptr_begin, void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
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
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
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
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
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
  if (!ompt_device_callbacks.is_tracing_enabled() ||
      (!ompt_device_callbacks.is_tracing_type_enabled(
           ompt_callback_target_data_op) &&
       !ompt_device_callbacks.is_tracing_type_enabled(
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
  ompt_device_callbacks.ompt_callback_target_submit_emi(
      ompt_scope_begin, &ompt_target_data, num_teams, opid_create,
      &ompt_target_region_opid);
}

void OmptInterface::target_submit_end(unsigned int num_teams) {
  ompt_device_callbacks.ompt_callback_target_submit_emi(
      ompt_scope_end, &ompt_target_data, num_teams, opid_get,
      &ompt_target_region_opid);
}

ompt_record_ompt_t *
OmptInterface::target_submit_trace_record_gen(unsigned int num_teams) {
  if (!ompt_device_callbacks.is_tracing_enabled() ||
      (!ompt_device_callbacks.is_tracing_type_enabled(
           ompt_callback_target_submit) &&
       !ompt_device_callbacks.is_tracing_type_enabled(
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
  ompt_device_callbacks.ompt_callback_target_emi(
      ompt_target_enter_data, ompt_scope_begin, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_create);
}

void OmptInterface::target_data_enter_end(int64_t device_id, void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_emi(
      ompt_target_enter_data, ompt_scope_end, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_get);
  target_region_end();
}

void OmptInterface::target_data_exit_begin(int64_t device_id, void *codeptr) {
  target_region_begin();
  ompt_device_callbacks.ompt_callback_target_emi(
      ompt_target_exit_data, ompt_scope_begin, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_create);
  target_region_announce("begin");
}

void OmptInterface::target_data_exit_end(int64_t device_id, void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_emi(
      ompt_target_exit_data, ompt_scope_end, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_get);
  target_region_end();
}

void OmptInterface::target_update_begin(int64_t device_id, void *codeptr) {
  target_region_begin();
  ompt_device_callbacks.ompt_callback_target_emi(
      ompt_target_update, ompt_scope_begin, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_create);
  target_region_announce("begin");
}

void OmptInterface::target_update_end(int64_t device_id, void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_emi(
      ompt_target_update, ompt_scope_end, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_get);
  target_region_end();
}

void OmptInterface::target_begin(int64_t device_id, void *codeptr) {
  target_region_begin();
  ompt_device_callbacks.ompt_callback_target_emi(
      ompt_target, ompt_scope_begin, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_create);
  target_region_announce("begin");
}

void OmptInterface::target_end(int64_t device_id, void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_emi(
      ompt_target, ompt_scope_end, device_id, ompt_task_data,
      ompt_target_task_data, &ompt_target_data, codeptr, regionid_get);
  target_region_end();
}

ompt_record_ompt_t *
OmptInterface::target_trace_record_gen(int64_t device_id, ompt_target_t kind,
                                       ompt_scope_endpoint_t endpoint,
                                       void *code) {
  if (!ompt_device_callbacks.is_tracing_enabled() ||
      (!ompt_device_callbacks.is_tracing_type_enabled(ompt_callback_target) &&
       !ompt_device_callbacks.is_tracing_type_enabled(
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

static int libomptarget_ompt_initialize(ompt_function_lookup_t lookup,
                                        int initial_device_num,
                                        ompt_data_t *tool_data) {
  DP("OMPT: enter libomptarget_ompt_initialize!\n");

  ompt_enabled = true;

#define ompt_bind_name(fn)                                                     \
  fn##_fn = (fn##_t)lookup(#fn);                                               \
  DP("%s=%p\n", #fn, fnptr_to_ptr(fn##_fn));

  ompt_bind_name(ompt_set_frame_enter);
  ompt_bind_name(ompt_get_task_data);
  ompt_bind_name(ompt_get_target_task_data);

#undef ompt_bind_name

  ompt_device_callbacks.register_callbacks(lookup);

  DP("OMPT: exit libomptarget_ompt_initialize!\n");

  return 0;
}

static void libomptarget_ompt_finalize(ompt_data_t *data) {
  DP("enter libomptarget_ompt_finalize!\n");

  libomptarget_rtl_finalizer.finalize();

  ompt_enabled = false;

  DP("exit libomptarget_ompt_finalize!\n");
}

// Today, this is not called from libomptarget
ompt_device *ompt_device_callbacks_t::lookup_device(int device_num) {
  assert(0 && "Lookup device should be invoked in the plugin");
  return nullptr;
}

ompt_interface_fn_t
ompt_device_callbacks_t::lookup(const char *interface_function_name) {
  if (strcmp(interface_function_name,
             stringify(LIBOMPTARGET_GET_TARGET_OPID)) == 0)
    return (ompt_interface_fn_t)LIBOMPTARGET_GET_TARGET_OPID;

  return ompt_device_callbacks.lookup_callback(interface_function_name);
}

#ifdef OMPT_SUPPORT
/*****************************************************************************
 * constructor
 *****************************************************************************/

void ompt_init() {
  DP("OMPT: Entering ompt_init\n");
  static library_ompt_connector_t libomp_connector("libomp");
  static ompt_start_tool_result_t ompt_result;

  ompt_result.initialize = libomptarget_ompt_initialize;
  ompt_result.finalize = libomptarget_ompt_finalize;
  ompt_result.tool_data.value = 0;

  ompt_device_callbacks.init();
  libomp_connector.connect(&ompt_result);

  DP("OMPT: Exit ompt_init\n");
}
#endif

extern "C" {

void libomptarget_ompt_connect(ompt_start_tool_result_t *result) {
  DP("OMPT: Enter libomptarget_ompt_connect: OMPT enabled == %d\n",
     ompt_enabled);
  if (ompt_enabled && result) {
    libomptarget_rtl_finalizer.register_rtl(result->finalize);
    result->initialize(ompt_device_callbacks_t::lookup, 0, NULL);
  }
  DP("OMPT: Leave libomptarget_ompt_connect\n");
}

// Device-independent entry point for ompt_set_trace_ompt
ompt_set_result_t libomptarget_ompt_set_trace_ompt(ompt_device_t *device,
                                                   unsigned int enable,
                                                   unsigned int etype) {
  std::unique_lock<std::mutex> lck(set_trace_mutex);
  return ompt_device_callbacks.set_trace_ompt(device, enable, etype);
}

// Device-independent entry point for ompt_start_trace
int libomptarget_ompt_start_trace(ompt_callback_buffer_request_t request,
                                  ompt_callback_buffer_complete_t complete) {
  std::unique_lock<std::mutex> lck(start_stop_flush_trace_mutex);
  ompt_device_callbacks.set_buffer_request(request);
  ompt_device_callbacks.set_buffer_complete(complete);
  if (request && complete) {
    ompt_device_callbacks.set_tracing_enabled(true);
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
  ompt_device_callbacks.set_tracing_enabled(false);
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
}
