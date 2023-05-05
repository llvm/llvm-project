//===---------- OmptCallback.cpp - Generic OMPT callbacks --------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OMPT support for PluginInterface
//
//===----------------------------------------------------------------------===//

#ifdef OMPT_SUPPORT
#include <atomic>
#include <cstdio>
#include <string.h>
#include <vector>

#include "Debug.h"
#include "ompt_connector.h"
#include "ompt_device_callbacks.h"

/// Object maintaining all the callbacks in the plugin
OmptDeviceCallbacksTy OmptDeviceCallbacks;

/// Lookup function used for querying callback functions maintained
/// by the plugin
ompt_interface_fn_t
OmptDeviceCallbacksTy::doLookup(const char *InterfaceFunctionName) {
  // TODO This will be populated with device tracing functions
  return (ompt_interface_fn_t) nullptr;
}

/// Used to indicate whether OMPT was enabled for this library
static bool OmptEnabled = false;

/// This function is passed to libomptarget as part of the OMPT connector
/// object. It is called by libomptarget during initialization of OMPT in the
/// plugin. \p lookup to be used to query callbacks registered with libomptarget
/// \p initial_device_num Initial device num provided by libomptarget
/// \p tool_data as provided by the tool
static int OmptDeviceInit(ompt_function_lookup_t lookup, int initial_device_num,
                          ompt_data_t *tool_data) {
  DP("OMPT: Enter OmptDeviceInit\n");
  OmptEnabled = true;
  // The lookup parameter is provided by libomptarget which already has the tool
  // callbacks registered at this point. The registration call below causes the
  // same callback functions to be registered in the plugin as well.
  OmptDeviceCallbacks.registerCallbacks(lookup);
  DP("OMPT: Exit OmptDeviceInit\n");
  return 0;
}

/// This function is passed to libomptarget as part of the OMPT connector
/// object. It is called by libomptarget during finalization of OMPT in the
/// plugin.
static void OmptDeviceFini(ompt_data_t *tool_data) {
  DP("OMPT: Executing OmptDeviceFini\n");
}

/// Used to initialize callbacks implemented by the tool. This interface will
/// lookup the callbacks table in libomptarget and assign them to the callbacks
/// table maintained in the calling plugin library.
void OmptCallbackInit() {
  DP("OMPT: Entering OmptCallbackInit\n");
  /// Connect plugin instance with libomptarget
  OmptLibraryConnectorTy LibomptargetConnector("libomptarget");
  ompt_start_tool_result_t OmptResult;

  // Initialize OmptResult with the init and fini functions that will be
  // called by the connector
  OmptResult.initialize = OmptDeviceInit;
  OmptResult.finalize = OmptDeviceFini;
  OmptResult.tool_data.value = 0;

  // Initialize the device callbacks first
  OmptDeviceCallbacks.init();

  // Now call connect that causes the above init/fini functions to be called
  LibomptargetConnector.connect(&OmptResult);
  DP("OMPT: Exiting OmptCallbackInit\n");
}
#endif
