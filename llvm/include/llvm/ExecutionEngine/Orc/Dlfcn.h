//===------ Dlfcn.h - Proxies for dlfcn-style calls -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Protocol-agnostic Proxy types for the ORC runtime's dlfcn-style calls, used
// to initialize, re-initialize and deinitialize JITDylibs in the executor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_DLFCN_H
#define LLVM_EXECUTIONENGINE_ORC_DLFCN_H

#include "llvm/ExecutionEngine/Orc/Proxy.h"

#include <cstdint>
#include <string>

namespace llvm::orc {

/// Open a JITDylib by name with the given dlopen-style mode flags; returns its
/// dso handle.
using DlfcnOpenProxy = Proxy<ExecutorAddr(std::string, int32_t)>;

/// Run newly added initializers for an already-open dso handle; returns nonzero
/// on failure.
using DlfcnUpdateProxy = Proxy<int32_t(ExecutorAddr)>;

/// Close the given dso handle, running its deinitializers; returns nonzero on
/// failure.
using DlfcnCloseProxy = Proxy<int32_t(ExecutorAddr)>;

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_DLFCN_H
