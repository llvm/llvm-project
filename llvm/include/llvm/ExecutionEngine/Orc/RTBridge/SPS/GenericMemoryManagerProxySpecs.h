//===- GenericMemoryManagerProxySpecs.h - SPS specs for mem mgmt -*- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS ProxySpecs for the GenericMemoryManagerProxies: each binds a Proxy to
// its controller-interface descriptor in
// Shared/SPSCI/SimpleNativeMemoryMapSPSCI.h, which supplies the wrapper name
// and wire signature. The names are the runtime's SimpleNativeMemoryMap
// defaults (the Create methods look symbols up under caller-supplied names).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_GENERICMEMORYMANAGERPROXYSPECS_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_GENERICMEMORYMANAGERPROXYSPECS_H

#include "llvm/ExecutionEngine/Orc/RTBridge/GenericMemoryManagerProxies.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/SPS/ProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/SimpleNativeMemoryMapSPSCI.h"

namespace llvm::orc::rt::sps {

using MemMgrReserveProxySpec =
    ProxySpec<rt::MemMgrReserveProxy, sps_ci::MemMgrReserve>;
using MemMgrInitializeProxySpec =
    ProxySpec<rt::MemMgrInitializeProxy, sps_ci::MemMgrInitialize>;
using MemMgrDeinitializeProxySpec =
    ProxySpec<rt::MemMgrDeinitializeProxy, sps_ci::MemMgrDeinitialize>;
using MemMgrReleaseProxySpec =
    ProxySpec<rt::MemMgrReleaseProxy, sps_ci::MemMgrRelease>;

} // namespace llvm::orc::rt::sps

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_GENERICMEMORYMANAGERPROXYSPECS_H
