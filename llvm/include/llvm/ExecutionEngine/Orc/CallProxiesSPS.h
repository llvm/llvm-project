//===----- CallProxiesSPS.h - SPS specs for CallProxies ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS ProxySpecs for the CallProxies: each binds a Proxy to its
// controller-interface descriptor in Shared/SPSCI/CallSPSCI.h, which supplies
// the wrapper name and wire signature.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_CALLPROXIESSPS_H
#define LLVM_EXECUTIONENGINE_ORC_CALLPROXIESSPS_H

#include "llvm/ExecutionEngine/Orc/CallProxies.h"
#include "llvm/ExecutionEngine/Orc/SPSProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/CallSPSCI.h"

namespace llvm::orc::sps {

/// SPS proxy for CallMainProxy: runs a main-like function
/// (int(int argc, char *argv[])) in the executor.
using CallMainProxySpec = ProxySpec<CallMainProxy, rt::sps_ci::CallMain>;

/// SPS proxy for CallVoidVoidProxy: runs a void() function in the executor.
/// WARNING: This Proxy is experimental and may be removed.
using CallVoidVoidProxySpec =
    ProxySpec<CallVoidVoidProxy, rt::sps_ci::CallVoidVoid>;

/// SPS proxy for CallInt32VoidProxy: runs an int32_t() function in the
/// executor.
/// WARNING: This Proxy is experimental and may be removed.
using CallInt32VoidProxySpec =
    ProxySpec<CallInt32VoidProxy, rt::sps_ci::CallInt32Void>;

/// SPS proxy for CallInt32Int32Proxy: runs an int32_t(int32_t) function in
/// the executor.
/// WARNING: This Proxy is experimental and may be removed.
using CallInt32Int32ProxySpec =
    ProxySpec<CallInt32Int32Proxy, rt::sps_ci::CallInt32Int32>;

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_CALLPROXIESSPS_H
