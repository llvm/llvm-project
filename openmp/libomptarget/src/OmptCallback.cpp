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

#include "llvm/Support/DynamicLibrary.h"

#include <cstdlib>
#include <cstring>
#include <memory>

#include "Debug.h"
#include "OmptCallback.h"
#include "OmptConnector.h"

using namespace llvm::omp::target::ompt;

// Define OMPT callback functions (bound to actual callbacks later on)
#define defineOmptCallback(Name, Type, Code)                                   \
  Name##_t llvm::omp::target::ompt::Name##_fn = nullptr;
FOREACH_OMPT_NOEMI_EVENT(defineOmptCallback)
FOREACH_OMPT_EMI_EVENT(defineOmptCallback)
#undef defineOmptCallback

/// Used to maintain the finalization functions that are received
/// from the plugins during connect.
/// Note: Currently, there are no plugin-specific finalizations, so each plugin
/// will call the same (empty) function.
class LibomptargetRtlFinalizer {
public:
  LibomptargetRtlFinalizer() {}

  void registerRtl(ompt_finalize_t FinalizationFunction) {
    if (FinalizationFunction) {
      RtlFinalizationFunctions.emplace_back(FinalizationFunction);
    }
  }

  void finalize() {
    for (auto FinalizationFunction : RtlFinalizationFunctions)
      FinalizationFunction(/* tool_data */ nullptr);
    RtlFinalizationFunctions.clear();
  }

private:
  llvm::SmallVector<ompt_finalize_t> RtlFinalizationFunctions;
};

/// Object that will maintain the RTL finalizer from the plugin
LibomptargetRtlFinalizer *LibraryFinalizer = nullptr;

ompt_get_callback_t llvm::omp::target::ompt::lookupCallbackByCode = nullptr;
ompt_function_lookup_t llvm::omp::target::ompt::lookupCallbackByName = nullptr;

int llvm::omp::target::ompt::initializeLibrary(ompt_function_lookup_t lookup,
                                               int initial_device_num,
                                               ompt_data_t *tool_data) {
  DP("OMPT: Executing initializeLibrary (libomp)\n");
#define bindOmptFunctionName(OmptFunction, DestinationFunction)                \
  DestinationFunction = (OmptFunction##_t)lookup(#OmptFunction);               \
  DP("OMPT: initializeLibrary (libomp) bound %s=%p\n", #DestinationFunction,   \
     ((void *)(uint64_t)DestinationFunction));

  bindOmptFunctionName(ompt_get_callback, lookupCallbackByCode);
#undef bindOmptFunctionName

  // Store pointer of 'ompt_libomp_target_fn_lookup' for use by libomptarget
  lookupCallbackByName = lookup;

  assert(lookupCallbackByCode && "lookupCallbackByCode should be non-null");
  assert(lookupCallbackByName && "lookupCallbackByName should be non-null");
  assert(LibraryFinalizer == nullptr &&
         "LibraryFinalizer should not be initialized yet");

  LibraryFinalizer = new LibomptargetRtlFinalizer();

  return 0;
}

void llvm::omp::target::ompt::finalizeLibrary(ompt_data_t *data) {
  DP("OMPT: Executing finalizeLibrary (libomp)\n");
  // Before disabling OMPT, call the (plugin) finalizations that were registered
  // with this library
  LibraryFinalizer->finalize();
  delete LibraryFinalizer;
}

void llvm::omp::target::ompt::connectLibrary() {
  DP("OMPT: Entering connectLibrary (libomp)\n");
  // Connect with libomp
  static OmptLibraryConnectorTy LibompConnector("libomp");
  static ompt_start_tool_result_t OmptResult;

  // Initialize OmptResult with the init and fini functions that will be
  // called by the connector
  OmptResult.initialize = ompt::initializeLibrary;
  OmptResult.finalize = ompt::finalizeLibrary;
  OmptResult.tool_data.value = 0;

  // Now call connect that causes the above init/fini functions to be called
  LibompConnector.connect(&OmptResult);

#define bindOmptCallback(Name, Type, Code)                                     \
  if (lookupCallbackByCode)                                                    \
    lookupCallbackByCode(                                                      \
        (ompt_callbacks_t)(Code),                                              \
        (ompt_callback_t *)&(llvm::omp::target::ompt::Name##_fn));
  FOREACH_OMPT_NOEMI_EVENT(bindOmptCallback)
  FOREACH_OMPT_EMI_EVENT(bindOmptCallback)
#undef bindOmptCallback

  DP("OMPT: Exiting connectLibrary (libomp)\n");
}

extern "C" {
/// Used for connecting libomptarget with a plugin
void ompt_libomptarget_connect(ompt_start_tool_result_t *result) {
  DP("OMPT: Enter ompt_libomptarget_connect\n");
  if (result && LibraryFinalizer) {
    // Cache each fini function, so that they can be invoked on exit
    LibraryFinalizer->registerRtl(result->finalize);
    // Invoke the provided init function with the lookup function maintained
    // in this library so that callbacks maintained by this library are
    // retrieved.
    result->initialize(lookupCallbackByName,
                       /* initial_device_num */ 0, /* tool_data */ nullptr);
  }
  DP("OMPT: Leave ompt_libomptarget_connect\n");
}
}
#else
extern "C" {
/// Dummy definition when OMPT is disabled
void ompt_libomptarget_connect() {}
}
#endif // OMPT_SUPPORT
