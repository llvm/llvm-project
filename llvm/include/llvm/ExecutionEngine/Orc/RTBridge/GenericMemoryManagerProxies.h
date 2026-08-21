//===- GenericMemoryManagerProxies.h - Proxies for mem mgmt -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Named rt::Proxy types for the executor's memory-manager operations. The
// instance address of the executor-side manager is passed as the first
// argument to each call.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_GENERICMEMORYMANAGERPROXIES_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_GENERICMEMORYMANAGERPROXIES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/Proxy.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>

namespace llvm::orc::rt {

/// Reserve an address range of the given size; returns its base.
using MemMgrReserveProxy =
    Proxy<Expected<ExecutorAddr>(ExecutorAddr, uint64_t)>;

/// Apply a finalize request; returns a key for the initialized allocation.
using MemMgrInitializeProxy =
    Proxy<Expected<ExecutorAddr>(ExecutorAddr, tpctypes::FinalizeRequest)>;

/// Deinitialize the allocations with the given base addresses (running their
/// deallocation actions) without releasing their memory.
using MemMgrDeinitializeProxy =
    Proxy<Error(ExecutorAddr, ArrayRef<ExecutorAddr>)>;

/// Release the allocations with the given base addresses.
using MemMgrReleaseProxy = Proxy<Error(ExecutorAddr, ArrayRef<ExecutorAddr>)>;

} // namespace llvm::orc::rt

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_GENERICMEMORYMANAGERPROXIES_H
