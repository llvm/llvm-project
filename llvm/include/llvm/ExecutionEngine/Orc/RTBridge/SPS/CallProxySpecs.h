//===----- CallProxySpecs.h - SPS specs for CallProxies ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS ProxySpecs (signatures, controller-interface names, and dispatch) for the
// CallProxies.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_CALLPROXYSPECS_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_CALLPROXYSPECS_H

#include "llvm/ExecutionEngine/Orc/RTBridge/CallProxies.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/SPS/ProxySpec.h"

#include <cstdint>

namespace llvm::orc::rt::sps {

using CallMainSPSSig = int64_t(shared::SPSExecutorAddr,
                               shared::SPSSequence<shared::SPSString>);
inline constexpr char CallMainCIName[] = "orc_rt_ci_sps_call_main";
/// SPS proxy for rt::CallMainProxy: runs a main-like function
/// (int(int argc, char *argv[])) in the executor.
using CallMainProxySpec =
    ProxySpec<rt::CallMainProxy, CallMainSPSSig, CallMainCIName>;

using CallVoidVoidSPSSig = void(shared::SPSExecutorAddr);
inline constexpr char CallVoidVoidCIName[] = "orc_rt_ci_sps_call_void_void";
/// SPS proxy for rt::CallVoidVoidProxy: runs a void() function in the executor.
/// WARNING: This Proxy is experimental and may be removed.
using CallVoidVoidProxySpec =
    ProxySpec<rt::CallVoidVoidProxy, CallVoidVoidSPSSig, CallVoidVoidCIName>;

using CallInt32VoidSPSSig = int32_t(shared::SPSExecutorAddr);
inline constexpr char CallInt32VoidCIName[] = "orc_rt_ci_sps_call_int32_void";
/// SPS proxy for rt::CallInt32VoidProxy: runs an int32_t() function in the
/// executor.
/// WARNING: This Proxy is experimental and may be removed.
using CallInt32VoidProxySpec =
    ProxySpec<rt::CallInt32VoidProxy, CallInt32VoidSPSSig, CallInt32VoidCIName>;

using CallInt32Int32SPSSig = int32_t(shared::SPSExecutorAddr, int32_t);
inline constexpr char CallInt32Int32CIName[] = "orc_rt_ci_sps_call_int32_int32";
/// SPS proxy for rt::CallInt32Int32Proxy: runs an int32_t(int32_t) function in
/// the executor.
/// WARNING: This Proxy is experimental and may be removed.
using CallInt32Int32ProxySpec =
    ProxySpec<rt::CallInt32Int32Proxy, CallInt32Int32SPSSig,
              CallInt32Int32CIName>;

} // namespace llvm::orc::rt::sps

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_CALLPROXYSPECS_H
