//===- EPCGenericJITLinkMemoryManagerSPS.h - SPS mem manager ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binds EPCGenericJITLinkMemoryManager to the ORC runtime's SPS controller
// interface: a ProxySpec per operation, plus factories that resolve them and
// construct an instance.
//
// Each spec pairs one of EPCGenericJITLinkMemoryManager's proxies with its
// controller-interface descriptor in Shared/SPSCI/SimpleNativeMemoryMapSPSCI.h,
// which supplies the wrapper name and wire signature. The specs are public so
// that clients can resolve the operations under non-default names, using
// recordProxy<Spec>(&P, Name) with lookupAndApply.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGERSPS_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGERSPS_H

#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/SPSProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/SimpleNativeMemoryMapSPSCI.h"
#include "llvm/Support/Compiler.h"

#include <memory>

namespace llvm::orc::sps {

using MemMgrReserveProxySpec =
    ProxySpec<EPCGenericJITLinkMemoryManager::ReserveProxy,
              rt::sps_ci::MemMgrReserve>;
using MemMgrInitializeProxySpec =
    ProxySpec<EPCGenericJITLinkMemoryManager::InitializeProxy,
              rt::sps_ci::MemMgrInitialize>;
using MemMgrDeinitializeProxySpec =
    ProxySpec<EPCGenericJITLinkMemoryManager::DeinitializeProxy,
              rt::sps_ci::MemMgrDeinitialize>;
using MemMgrReleaseProxySpec =
    ProxySpec<EPCGenericJITLinkMemoryManager::ReleaseProxy,
              rt::sps_ci::MemMgrRelease>;

/// Create an EPCGenericJITLinkMemoryManager for the ORC runtime's
/// SimpleNativeMemoryMap interface, resolving its symbols in the given
/// JITDylib.
LLVM_ABI Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
createEPCGenericJITLinkMemoryManager(JITDylib &JD);

/// Create an EPCGenericJITLinkMemoryManager for the ORC runtime's
/// SimpleNativeMemoryMap interface, resolving its symbols in the given
/// ExecutionSession's bootstrap JITDylib.
LLVM_ABI Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
createEPCGenericJITLinkMemoryManager(ExecutionSession &ES);

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGERSPS_H
