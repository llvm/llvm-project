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

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdlib>
#include <cstring>
#include <memory>

#include "Shared/Debug.h"

#include "OpenMP/OMPT/Callback.h"
#include "OpenMP/OMPT/Connector.h"

using namespace llvm::omp::target::ompt;

bool llvm::omp::target::ompt::CallbacksInitialized = false;

ompt_get_callback_t llvm::omp::target::ompt::lookupCallbackByCode = nullptr;
ompt_function_lookup_t llvm::omp::target::ompt::lookupCallbackByName = nullptr;

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

int llvm::omp::target::ompt::initializeLibrary(ompt_function_lookup_t lookup,
                                               int initial_device_num,
                                               ompt_data_t *tool_data) {
  DP("Executing initializeLibrary (libomptarget)\n");

#define bindOmptFunctionName(OmptFunction, DestinationFunction)                \
  if (lookup)                                                                  \
    DestinationFunction = (OmptFunction##_t)lookup(#OmptFunction);             \
  DP("initializeLibrary (libomptarget) bound %s=%p\n", #DestinationFunction,   \
     ((void *)(uint64_t)DestinationFunction));

  bindOmptFunctionName(ompt_get_callback, lookupCallbackByCode);
#undef bindOmptFunctionName

    // Store pointer of 'ompt_libomp_target_fn_lookup' for use by the plugin
    lookupCallbackByName = lookup;

    CallbacksInitialized = true;

    return 0;
}

void llvm::omp::target::ompt::finalizeLibrary(ompt_data_t *tool_data) {
  DP("Executing finalizeLibrary (libomptarget)\n");
}

void llvm::omp::target::ompt::connectLibrary() {
  DP("Entering connectLibrary (libomptarget)\n");
  /// Connect plugin instance with libomptarget
  OmptLibraryConnectorTy LibomptargetConnector("libomptarget");
  ompt_start_tool_result_t OmptResult;

  // Initialize OmptResult with the init and fini functions that will be
  // called by the connector
  OmptResult.initialize = ompt::initializeLibrary;
  OmptResult.finalize = ompt::finalizeLibrary;
  OmptResult.tool_data.value = 0;

  // Now call connect that causes the above init/fini functions to be called
  LibomptargetConnector.connect(&OmptResult);
  DP("Exiting connectLibrary (libomptarget)\n");
}

#pragma pop_macro("DEBUG_PREFIX")

#endif // OMPT_SUPPORT
