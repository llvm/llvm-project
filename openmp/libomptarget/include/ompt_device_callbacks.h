//===--------- ompt_device_callbacks.h - OMPT callbacks -- C++ ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface used by both target-independent and device-dependent runtimes
// to coordinate registration and invocation of OMPT callbacks
//
//===----------------------------------------------------------------------===//

#ifndef _OMPT_DEVICE_CALLBACKS_H
#define _OMPT_DEVICE_CALLBACKS_H

#ifdef OMPT_SUPPORT

#include "Debug.h"
#include <omp-tools.h>

#define DEBUG_PREFIX "OMPT"

#define FOREACH_OMPT_TARGET_CALLBACK(macro)                                    \
  FOREACH_OMPT_DEVICE_EVENT(macro)                                             \
  FOREACH_OMPT_NOEMI_EVENT(macro)                                              \
  FOREACH_OMPT_EMI_EVENT(macro)

/// Internal representation for OMPT device callback functions.
class OmptDeviceCallbacksTy {
public:
  /// Initialize the enabled flag and all the callbacks
  void init() {
    Enabled = false;
#define initName(Name, Type, Code) Name##_fn = 0;
    FOREACH_OMPT_TARGET_CALLBACK(initName)
#undef initName
  }

  /// Used to register callbacks. \p Lookup is used to query a given callback
  /// by name and the result is assigned to the corresponding callback function.
  void registerCallbacks(ompt_function_lookup_t Lookup) {
    Enabled = true;
#define OmptBindCallback(Name, Type, Code)                                     \
  Name##_fn = (Name##_t)Lookup(#Name);                                         \
  DP("OMPT: class bound %s=%p\n", #Name, ((void *)(uint64_t)Name##_fn));

    FOREACH_OMPT_TARGET_CALLBACK(OmptBindCallback);
#undef OmptBindCallback
  }

private:
  /// Set to true if callbacks for this library have been initialized
  bool Enabled;

  /// Callback functions
#define DeclareName(Name, Type, Code) Name##_t Name##_fn;
  FOREACH_OMPT_TARGET_CALLBACK(DeclareName)
#undef DeclareName
};

/// Device callbacks object for the library that performs the instantiation
extern OmptDeviceCallbacksTy OmptDeviceCallbacks;

#undef DEBUG_PREFIX

#endif // OMPT_SUPPORT

#endif // _OMPT_DEVICE_CALLBACKS_H
