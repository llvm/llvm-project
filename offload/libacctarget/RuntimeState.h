//===-------- RuntimeState.h -------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBACCTARGET_RUNTIMESTATE_H_
#define _LIBACCTARGET_RUNTIMESTATE_H_

#include "PluginManager.h"

namespace llvm::acc::target {
extern PluginManager *PM;
} // namespace llvm::acc::target

/// Initialize the OpenACC runtime.
void initAccRuntime();

/// Completely deinitialize the OpenACC runtime.
void deinitAccRuntime();

#endif // _LIBACCTARGET_RUNTIMESTATE_H_
