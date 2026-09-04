//===----------------------- OrcRTBootstrap.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OrcRTBootstrap provides functions that should be linked into the executor
// to bootstrap common JIT functionality (e.g. memory allocation and memory
// access).
//
// Call addTo to add these functions to a bootstrap symbols map.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_ORCRTBOOTSTRAP_H
#define LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_ORCRTBOOTSTRAP_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"

namespace llvm::orc::rt_bootstrap {

/// Adds executor-side wrappers for the run-as function proxies.
void addRunAsFunctionWrappersTo(StringMap<ExecutorAddr> &M);

/// Adds all default target-process bootstrap wrappers.
void addTo(StringMap<ExecutorAddr> &M);

} // namespace llvm::orc::rt_bootstrap

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_ORCRTBOOTSTRAP_H
