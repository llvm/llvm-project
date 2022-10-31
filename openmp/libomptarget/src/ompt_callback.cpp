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
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

#include "omp-tools.h"
#include "ompt-connector.h"
#include "private.h"

#define fnptr_to_ptr(x) ((void *)(uint64_t)x)

/// Used to indicate whether OMPT was enabled for this library
bool ompt_enabled = false;

#ifdef OMPT_SUPPORT
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
  ompt_enabled = true;
  // TODO use the parameters to populate callbacks in libomptarget
  DP("exit ompt_libomptarget_initialize!\n");
  return 0;
}

static void ompt_libomptarget_finalize(ompt_data_t *data) {
  DP("enter ompt_libomptarget_finalize!\n");
  ompt_enabled = false;
  DP("exit ompt_libomptarget_finalize!\n");
}

/*****************************************************************************
 * constructor
 *****************************************************************************/
/// Used to initialize callbacks implemented by the tool. This interface
/// will lookup the callbacks table in libomp and assign them to the callbacks
/// maintained in libomptarget. Using priority 102 to have this constructor
/// run after the init target library constructor with priority 101 (see
/// rtl.cpp).
__attribute__((constructor(102))) static void ompt_init(void) {
  DP("OMPT: Enter ompt_init\n");
  // Connect with libomp
  static OmptLibraryConnectorTy LibompConnector("ompt_libomp");
  static ompt_start_tool_result_t OmptResult;

  // Initialize OmptResult with the init and fini functions that will be
  // called by the connector
  OmptResult.initialize = ompt_libomptarget_initialize;
  OmptResult.finalize = ompt_libomptarget_finalize;
  OmptResult.tool_data.value = 0;

  // Now call connect that causes the above init/fini functions to be called
  LibompConnector.connect(&OmptResult);
  DP("OMPT: Exit ompt_init\n");
}
#endif
