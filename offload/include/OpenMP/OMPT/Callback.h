//===-- OpenMP/OMPT/Callback.h - OpenMP Tooling callbacks -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface used by target-independent runtimes to coordinate registration and
// invocation of OMPT callbacks and initialization / finalization.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_INCLUDE_OPENMP_OMPT_CALLBACK_H
#define OFFLOAD_INCLUDE_OPENMP_OMPT_CALLBACK_H

#ifdef OMPT_SUPPORT

#include "omp-tools.h"

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

#define FOREACH_OMPT_TARGET_CALLBACK(macro)                                    \
  FOREACH_OMPT_DEVICE_EVENT(macro)                                             \
  FOREACH_OMPT_NOEMI_EVENT(macro)                                              \
  FOREACH_OMPT_EMI_EVENT(macro)

#define performIfOmptInitialized(stmt)                                         \
  do {                                                                         \
    if (llvm::omp::target::ompt::Initialized) {                                \
      stmt;                                                                    \
    }                                                                          \
  } while (0)

#define performOmptCallback(CallbackName, ...)                                 \
  do {                                                                         \
    if (ompt_callback_##CallbackName##_fn)                                     \
      ompt_callback_##CallbackName##_fn(__VA_ARGS__);                          \
  } while (0)

/// Function type def used for maintaining unique target region, target
/// operations ids
typedef uint64_t (*IdInterfaceTy)();

namespace llvm {
namespace omp {
namespace target {
namespace ompt {

#define declareOmptCallback(Name, Type, Code) extern Name##_t Name##_fn;
FOREACH_OMPT_NOEMI_EVENT(declareOmptCallback)
FOREACH_OMPT_EMI_EVENT(declareOmptCallback)
#undef declareOmptCallback

/// This function will call an OpenMP API function. Which in turn will lookup a
/// given enum value of type \p ompt_callbacks_t and copy the address of the
/// corresponding callback funtion into the provided pointer.
/// The pointer to the runtime function is passed during 'initializeLibrary'.
/// \p which the enum value of the requested callback function
/// \p callback the destination pointer where the address shall be copied
extern ompt_get_callback_t lookupCallbackByCode;

/// Lookup function to be used by the lower layer (e.g. the plugin). This
/// function has to be provided when actually calling callback functions like
/// 'ompt_callback_device_initialize_fn' (param: 'lookup').
/// The pointer to the runtime function is passed during 'initializeLibrary'.
/// \p InterfaceFunctionName the name of the OMPT callback function to look up
extern ompt_function_lookup_t lookupCallbackByName;

/// This is the function called by the higher layer (libomp / libomtarget)
/// responsible for initializing OMPT in this library. This is passed to libomp
/// as part of the OMPT connector object.
/// \p lookup to be used to query callbacks registered with libomp
/// \p initial_device_num initial device num (id) provided by libomp
/// \p tool_data as provided by the tool
int initializeLibrary(ompt_function_lookup_t lookup, int initial_device_num,
                      ompt_data_t *tool_data);

/// This function is passed to libomp / libomtarget as part of the OMPT
/// connector object. It is called by libomp during finalization of OMPT in
/// libomptarget -OR- by libomptarget during finalization of OMPT in the plugin.
/// \p tool_data as provided by the tool
void finalizeLibrary(ompt_data_t *tool_data);

/// This function will connect the \p initializeLibrary and \p finalizeLibrary
/// functions to their respective higher layer.
void connectLibrary();

/// OMPT initialization status; false if initializeLibrary has not been executed
extern bool Initialized;

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm

#pragma pop_macro("DEBUG_PREFIX")

#else
#define performIfOmptInitialized(stmt)
#endif // OMPT_SUPPORT

#endif // OFFLOAD_INCLUDE_OPENMP_OMPT_CALLBACK_H
