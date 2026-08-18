//===- SimpleMemoryMapSPS.h - SPS memory-map bindings -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binds SimpleMemoryMapBindings to the ORC runtime's SPS controller
// interface: a ProxySpec per operation, plus operations that resolve them.
//
// Each spec pairs one of the bindings' proxies with its controller-interface
// descriptor in Shared/SPSCI/SimpleNativeMemoryMapSPSCI.h, which supplies the
// wrapper name and wire signature. The specs are public so that clients can
// resolve the operations under non-default names, using
// recordProxy<Spec>(&P, Name) with lookupAndApply.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SIMPLEMEMORYMAPSPS_H
#define LLVM_EXECUTIONENGINE_ORC_SIMPLEMEMORYMAPSPS_H

#include "llvm/ExecutionEngine/Orc/LookupAndApply.h"
#include "llvm/ExecutionEngine/Orc/SPSProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/SimpleNativeMemoryMapSPSCI.h"
#include "llvm/ExecutionEngine/Orc/SimpleMemoryMap.h"
#include "llvm/Support/Compiler.h"

namespace llvm::orc::sps {

using MemMgrReserveProxySpec =
    ProxySpec<SimpleMemoryMapBindings::ReserveProxy, rt::sps_ci::MemMgrReserve>;
using MemMgrInitializeProxySpec =
    ProxySpec<SimpleMemoryMapBindings::InitializeProxy,
              rt::sps_ci::MemMgrInitialize>;
using MemMgrDeinitializeProxySpec =
    ProxySpec<SimpleMemoryMapBindings::DeinitializeProxy,
              rt::sps_ci::MemMgrDeinitialize>;
using MemMgrReleaseProxySpec =
    ProxySpec<SimpleMemoryMapBindings::ReleaseProxy, rt::sps_ci::MemMgrRelease>;

/// Build bindings over the SPS controller interface, resolving the operations
/// in the given JITDylib under the specs' default (SimpleNativeMemoryMap)
/// names.
///
/// To bind a different executor-side implementation, use the specs above with
/// recordProxy<Spec>(&P, Name) to resolve its own names instead.
LLVM_ABI Expected<SimpleMemoryMapBindings>
createSimpleMemoryMapBindings(JITDylib &JD);

/// As above, resolving the operations in ES's bootstrap JITDylib.
LLVM_ABI Expected<SimpleMemoryMapBindings>
createSimpleMemoryMapBindings(ExecutionSession &ES);

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_SIMPLEMEMORYMAPSPS_H
