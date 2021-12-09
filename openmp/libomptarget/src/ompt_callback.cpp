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
#include <dlfcn.h>

//****************************************************************************
// local include files
//****************************************************************************

#include <omp-tools.h>

#include "ompt_callback.h"
#include "private.h"

#include <ompt-connector.h>
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

/*****************************************************************************
 * private data
 *****************************************************************************/

static ompt_set_frame_enter_t ompt_set_frame_enter_fn = 0;
static ompt_get_task_data_t ompt_get_task_data_fn = 0;
static ompt_get_target_task_data_t ompt_get_target_task_data_fn = 0;

static libomptarget_rtl_finalizer_t libomptarget_rtl_finalizer;

const char *ompt_device_callbacks_t::documentation = 0;

/*****************************************************************************
 * Thread local data
 *****************************************************************************/

thread_local OmptInterface ompt_interface;

static thread_local uint64_t ompt_target_region_opid = 1;
static thread_local ompt_data_t ompt_target_data = ompt_data_none;
static thread_local ompt_data_t *ompt_task_data = 0;
static thread_local ompt_data_t *ompt_target_task_data = 0;
static thread_local ompt_id_t host_op_id = 0;

static std::atomic<uint64_t> unique_id_ticket(1);

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

  *ompt_task_data = ompt_data_none;
  *ompt_target_task_data = ompt_data_none;
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
      ompt_target_data_alloc, hst_ptr_begin, device_id, NULL, 0, size, codeptr,
      opid_create, &ompt_target_region_opid);
  target_operation_begin();
}

void OmptInterface::target_data_alloc_end(int64_t device_id,
                                          void *hst_ptr_begin, size_t size,
                                          void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
      ompt_scope_end, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_alloc, hst_ptr_begin, device_id, NULL, 0, size, codeptr,
      opid_get, &ompt_target_region_opid);
  target_operation_end();
}

void OmptInterface::target_data_submit_begin(int64_t device_id,
                                             void *tgt_ptr_begin,
                                             void *hst_ptr_begin, size_t size,
                                             void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
      ompt_scope_begin, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_transfer_to_device, hst_ptr_begin, 0, tgt_ptr_begin,
      device_id, size, codeptr, opid_create, &ompt_target_region_opid);
  target_operation_begin();
}

void OmptInterface::target_data_submit_end(int64_t device_id,
                                           void *tgt_ptr_begin,
                                           void *hst_ptr_begin, size_t size,
                                           void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
      ompt_scope_end, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_transfer_to_device, hst_ptr_begin, 0, tgt_ptr_begin,
      device_id, size, codeptr, opid_get, &ompt_target_region_opid);
  target_operation_end();
}

void OmptInterface::target_data_delete_begin(int64_t device_id,
                                             void *tgt_ptr_begin,
                                             void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
      ompt_scope_begin, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_delete, tgt_ptr_begin, device_id, NULL, 0, 0, codeptr,
      opid_create, &ompt_target_region_opid);
  target_operation_begin();
}

void OmptInterface::target_data_delete_end(int64_t device_id,
                                           void *tgt_ptr_begin, void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
      ompt_scope_end, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_delete, tgt_ptr_begin, device_id, NULL, 0, 0, codeptr,
      opid_get, &ompt_target_region_opid);
  target_operation_end();
}

void OmptInterface::target_data_retrieve_begin(int64_t device_id,
                                               void *hst_ptr_begin,
                                               void *tgt_ptr_begin, size_t size,
                                               void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
      ompt_scope_begin, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_transfer_from_device, tgt_ptr_begin, device_id,
      hst_ptr_begin, 0, size, codeptr, opid_create, &ompt_target_region_opid);
  target_operation_begin();
}

void OmptInterface::target_data_retrieve_end(int64_t device_id,
                                             void *hst_ptr_begin,
                                             void *tgt_ptr_begin, size_t size,
                                             void *codeptr) {
  ompt_device_callbacks.ompt_callback_target_data_op_emi(
      ompt_scope_end, ompt_target_task_data, &ompt_target_data,
      ompt_target_data_transfer_from_device, tgt_ptr_begin, device_id,
      hst_ptr_begin, 0, size, codeptr, opid_get, &ompt_target_region_opid);
  target_operation_end();
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
  DP("enter libomptarget_ompt_initialize!\n");

  ompt_enabled = true;

#define ompt_bind_name(fn)                                                     \
  fn##_fn = (fn##_t)lookup(#fn);                                               \
  DP("%s=%p\n", #fn, fnptr_to_ptr(fn##_fn));

  ompt_bind_name(ompt_set_frame_enter);
  ompt_bind_name(ompt_get_task_data);
  ompt_bind_name(ompt_get_target_task_data);

#undef ompt_bind_name

  ompt_device_callbacks.register_callbacks(lookup);

  DP("exit libomptarget_ompt_initialize!\n");

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

/*****************************************************************************
 * constructor
 *****************************************************************************/

__attribute__((constructor(102))) static void ompt_init(void) {
  static library_ompt_connector_t libomp_connector("libomp");
  static ompt_start_tool_result_t ompt_result;

  ompt_result.initialize = libomptarget_ompt_initialize;
  ompt_result.finalize = libomptarget_ompt_finalize;
  ompt_result.tool_data.value = 0;

  ompt_device_callbacks.init();

  libomp_connector.connect(&ompt_result);
  DP("OMPT: Exit ompt_init\n");
}

extern "C" {

void libomptarget_ompt_connect(ompt_start_tool_result_t *result) {
  DP("OMPT: Enter libomptarget_ompt_connect\n");
  if (ompt_enabled && result) {
    libomptarget_rtl_finalizer.register_rtl(result->finalize);
    result->initialize(ompt_device_callbacks_t::lookup, 0, NULL);
  }
  DP("OMPT: Leave libomptarget_ompt_connect\n");
}
}
