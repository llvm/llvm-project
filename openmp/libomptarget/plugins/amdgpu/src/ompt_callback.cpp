//===------ ompt_callback.cpp - Target RTLs Implementation -------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OMPT support for AMDGPU
//
//===----------------------------------------------------------------------===//

//****************************************************************************
// global includes
//****************************************************************************

#include <atomic>
#include <vector>

#include <string.h>

//****************************************************************************
// debug macro needed by include files
//****************************************************************************

#ifndef DEBUG_PREFIX
#define DEBUG_PREFIX "Target AMDGPU RTL"
#endif

//****************************************************************************
// local includes
//****************************************************************************

#include <Debug.h>
#include <ompt-connector.h>
#include <ompt_device_callbacks.h>

//****************************************************************************
// macros
//****************************************************************************

#define FOREACH_TARGET_FN(macro)

#define fnptr_to_ptr(x) ((void *)(uint64_t)x)

#define ompt_ptr_unknown ((void *)0)

//****************************************************************************
// global data
//****************************************************************************

ompt_device_callbacks_t ompt_device_callbacks;

//****************************************************************************
// private data
//****************************************************************************

static bool ompt_enabled = false;

static ompt_get_target_info_t LIBOMPTARGET_GET_TARGET_OPID;

const char *ompt_device_callbacks_t::documentation = 0;

static ompt_device *devices = 0;

//****************************************************************************
// private operations
//****************************************************************************

void ompt_device_callbacks_t::resize(int number_of_devices) {
  devices = new ompt_device[number_of_devices];
}

ompt_device *ompt_device_callbacks_t::lookup_device(int device_num) {
  return &devices[device_num];
}

ompt_interface_fn_t
ompt_device_callbacks_t::lookup(const char *interface_function_name) {
#define macro(fn)                                                              \
  if (strcmp(interface_function_name, #fn) == 0)                               \
    return (ompt_interface_fn_t)fn;

  FOREACH_TARGET_FN(macro);

#undef macro

  return (ompt_interface_fn_t)0;
}

static int ompt_device_init(ompt_function_lookup_t lookup,
                            int initial_device_num, ompt_data_t *tool_data) {
  DP("OMPT: Enter ompt_device_init\n");

  ompt_enabled = true;

  LIBOMPTARGET_GET_TARGET_OPID =
      (ompt_get_target_info_t)lookup(stringify(LIBOMPTARGET_GET_TARGET_OPID));

  DP("OMPT: libomptarget_get_target_info = %p\n",
     fnptr_to_ptr(LIBOMPTARGET_GET_TARGET_OPID));

  ompt_device_callbacks.register_callbacks(lookup);

  DP("OMPT: Exit ompt_device_init\n");

  return 0;
}

static void ompt_device_fini(ompt_data_t *tool_data) {
  DP("OMPT: executing amdgpu_ompt_device_fini\n");
}

//****************************************************************************
// constructor
//****************************************************************************

__attribute__((constructor)) static void ompt_init(void) {
  DP("OMPT: Entering ompt_init\n");
  static library_ompt_connector_t libomptarget_connector("libomptarget");
  static ompt_start_tool_result_t ompt_result;

  ompt_result.initialize = ompt_device_init;
  ompt_result.finalize = ompt_device_fini;
  ompt_result.tool_data.value = 0;
  ;

  ompt_device_callbacks.init();

  libomptarget_connector.connect(&ompt_result);
  DP("OMPT: Exiting ompt_init\n");
}
