//===- SharedMemoryMapSPS.h - SPS shared-memory map bindings ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binds SharedMemoryMapBindings to the ORC runtime's SPS controller interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHAREDMEMORYMAPSPS_H
#define LLVM_EXECUTIONENGINE_ORC_SHAREDMEMORYMAPSPS_H

#include "llvm/ExecutionEngine/Orc/LookupAndApply.h"
#include "llvm/ExecutionEngine/Orc/SPSProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/SharedMemoryMapperSPSCI.h"
#include "llvm/ExecutionEngine/Orc/SharedMemoryMap.h"
#include "llvm/Support/Compiler.h"

namespace llvm::orc::sps {

/// A ProxySpec per operation, pairing one of the bindings' proxies with its
/// controller-interface descriptor in Shared/SPSCI/SharedMemoryMapperSPSCI.h,
/// which supplies the wrapper name and wire signature. The specs are public so
/// that clients can resolve the operations under non-default names, using
/// recordProxy<Spec>(&P, Name) with lookupAndApply.
using SharedMemoryMapReserveProxySpec =
    ProxySpec<SharedMemoryMapBindings::ReserveProxy,
              rt::sps_ci::SharedMemoryMapperReserve>;
using SharedMemoryMapInitializeProxySpec =
    ProxySpec<SharedMemoryMapBindings::InitializeProxy,
              rt::sps_ci::SharedMemoryMapperInitialize>;
using SharedMemoryMapDeinitializeProxySpec =
    ProxySpec<SharedMemoryMapBindings::DeinitializeProxy,
              rt::sps_ci::SharedMemoryMapperDeinitialize>;
using SharedMemoryMapReleaseProxySpec =
    ProxySpec<SharedMemoryMapBindings::ReleaseProxy,
              rt::sps_ci::SharedMemoryMapperRelease>;

/// Build bindings over the SPS controller interface, resolving the operations
/// in the given JITDylib under the specs' default names.
LLVM_ABI Expected<SharedMemoryMapBindings>
createSharedMemoryMapBindings(JITDylib &JD);

/// As above, resolving the operations in ES's bootstrap JITDylib.
LLVM_ABI Expected<SharedMemoryMapBindings>
createSharedMemoryMapBindings(ExecutionSession &ES);

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_SHAREDMEMORYMAPSPS_H
