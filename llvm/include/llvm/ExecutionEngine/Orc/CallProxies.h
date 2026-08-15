//===------- CallProxies.h - Proxies for running functions ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Named Proxy types for running functions in the executor. Each takes the
// target function's ExecutorAddr (plus any arguments) and runs it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_CALLPROXIES_H
#define LLVM_EXECUTIONENGINE_ORC_CALLPROXIES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ExecutionEngine/Orc/Proxy.h"

#include <cstdint>
#include <string>

namespace llvm::orc {

/// Protocol-agnostic interface for running a main-like function
/// (int(int argc, char *argv[])) in the executor.
///
/// The function to run is given by its ExecutorAddr, its arguments as an
/// argument vector, and its int64_t result is returned.
using CallMainProxy = Proxy<int64_t(ExecutorAddr, ArrayRef<std::string>)>;

/// Protocol-agnostic interface for running a void() function in the executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Proxy is experimental and may be removed.
using CallVoidVoidProxy = Proxy<void(ExecutorAddr)>;

/// Protocol-agnostic interface for running an int32_t() function in the
/// executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Proxy is experimental and may be removed.
using CallInt32VoidProxy = Proxy<int32_t(ExecutorAddr)>;

/// Protocol-agnostic interface for running an int32_t(int32_t) function in the
/// executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Proxy is experimental and may be removed.
using CallInt32Int32Proxy = Proxy<int32_t(ExecutorAddr, int32_t)>;

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_CALLPROXIES_H
