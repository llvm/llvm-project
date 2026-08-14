//===- GenericMemoryManagerProxySpecs.h - SPS specs for mem mgmt -*- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS ProxySpecs (signatures, controller-interface names, and dispatch) for the
// GenericMemoryManagerProxies. The controller-interface names here are the
// runtime's SimpleNativeMemoryMap defaults (the Create methods look symbols up
// under caller-supplied names).
//
// The signatures below duplicate the SPSSimpleExecutorMemoryManager* signatures
// in OrcRTBridge.h; the intent is to retire those and have callers depend on
// this header instead.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_GENERICMEMORYMANAGERPROXYSPECS_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_GENERICMEMORYMANAGERPROXYSPECS_H

#include "llvm/ExecutionEngine/Orc/RTBridge/GenericMemoryManagerProxies.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/SPS/ProxySpec.h"

#include <cstdint>

namespace llvm::orc::rt::sps {

// Controller-interface name of the SimpleNativeMemoryMap instance: a data
// symbol (the allocator object) passed as the first argument to each call, not
// a wrapper to proxy.
inline constexpr char MemMgrInstanceCIName[] =
    "orc_rt_ci_SimpleNativeMemoryMap_Instance";

using MemMgrReserveSPSSig = shared::SPSExpected<shared::SPSExecutorAddr>(
    shared::SPSExecutorAddr, uint64_t);
inline constexpr char MemMgrReserveCIName[] =
    "orc_rt_ci_sps_SimpleNativeMemoryMap_reserve";
using MemMgrReserveProxySpec =
    ProxySpec<rt::MemMgrReserveProxy, MemMgrReserveSPSSig, MemMgrReserveCIName>;

using MemMgrInitializeSPSSig = shared::SPSExpected<shared::SPSExecutorAddr>(
    shared::SPSExecutorAddr, shared::SPSFinalizeRequest);
inline constexpr char MemMgrInitializeCIName[] =
    "orc_rt_ci_sps_SimpleNativeMemoryMap_initialize";
using MemMgrInitializeProxySpec =
    ProxySpec<rt::MemMgrInitializeProxy, MemMgrInitializeSPSSig,
              MemMgrInitializeCIName>;

using MemMgrDeinitializeSPSSig = shared::SPSError(
    shared::SPSExecutorAddr, shared::SPSSequence<shared::SPSExecutorAddr>);
inline constexpr char MemMgrDeinitializeCIName[] =
    "orc_rt_ci_sps_SimpleNativeMemoryMap_deinitializeMultiple";
using MemMgrDeinitializeProxySpec =
    ProxySpec<rt::MemMgrDeinitializeProxy, MemMgrDeinitializeSPSSig,
              MemMgrDeinitializeCIName>;

using MemMgrReleaseSPSSig = shared::SPSError(
    shared::SPSExecutorAddr, shared::SPSSequence<shared::SPSExecutorAddr>);
inline constexpr char MemMgrReleaseCIName[] =
    "orc_rt_ci_sps_SimpleNativeMemoryMap_releaseMultiple";
using MemMgrReleaseProxySpec =
    ProxySpec<rt::MemMgrReleaseProxy, MemMgrReleaseSPSSig, MemMgrReleaseCIName>;

} // namespace llvm::orc::rt::sps

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_GENERICMEMORYMANAGERPROXYSPECS_H
