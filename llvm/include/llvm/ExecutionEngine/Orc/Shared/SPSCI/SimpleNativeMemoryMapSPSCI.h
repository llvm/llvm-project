//===-- SimpleNativeMemoryMapSPSCI.h - SPS CI for mem mgmt ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS controller-interface descriptors for the runtime's SimpleNativeMemoryMap
// interface. Instances of the executor-side manager are created by calling
// MemMgrCreate, and the instance address is passed as the first argument to
// each of the other calls.
//
// The names below are the SimpleNativeMemoryMap defaults. Callers may resolve
// these operations under other names, so a descriptor's Name is a default
// rather than a fixed part of the contract.
//
// See CallSPSCI.h for a description of the descriptor scheme.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_SIMPLENATIVEMEMORYMAPSPSCI_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_SIMPLENATIVEMEMORYMAPSPSCI_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>

namespace llvm::orc::rt::sps_ci {

/// Create a new executor-side memory-manager instance; returns its address.
/// The instance is owned by the executor, and lives until the executor shuts
/// down.
struct MemMgrCreate {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("orc_rt_ci_sps_SimpleNativeMemoryMap_create");
  using SPSSig = shared::SPSExpected<shared::SPSExecutorAddr>();
};

/// A pre-created executor-side memory-manager instance. This is a data symbol
/// (the allocator object), not a wrapper to call.
///
/// This is published by executors that do not provide MemMgrCreate --
/// currently only LLVM's OrcTargetProcess library, whose wrapper functions
/// have no context through which a new instance could be registered. Executors
/// built on the ORC runtime provide MemMgrCreate instead.
///
/// FIXME: Remove this once OrcTargetProcess is retired.
inline constexpr SymbolNameSpec SimpleNativeMemoryMapInstanceName =
    SymbolNameSpec::c("orc_rt_ci_SimpleNativeMemoryMap_Instance");

/// Reserve an address range of the given size; returns its base.
struct MemMgrReserve {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("orc_rt_ci_sps_SimpleNativeMemoryMap_reserve");
  using SPSSig = shared::SPSExpected<shared::SPSExecutorAddr>(
      shared::SPSExecutorAddr, uint64_t);
};

/// Apply a finalize request; returns a key for the initialized allocation.
struct MemMgrInitialize {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("orc_rt_ci_sps_SimpleNativeMemoryMap_initialize");
  using SPSSig = shared::SPSExpected<shared::SPSExecutorAddr>(
      shared::SPSExecutorAddr, shared::SPSFinalizeRequest);
};

/// Deinitialize the allocations with the given base addresses (running their
/// deallocation actions) without releasing their memory.
struct MemMgrDeinitialize {
  static constexpr SymbolNameSpec Name = SymbolNameSpec::c(
      "orc_rt_ci_sps_SimpleNativeMemoryMap_deinitializeMultiple");
  using SPSSig = shared::SPSError(shared::SPSExecutorAddr,
                                  shared::SPSSequence<shared::SPSExecutorAddr>);
};

/// Release the allocations with the given base addresses.
struct MemMgrRelease {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("orc_rt_ci_sps_SimpleNativeMemoryMap_releaseMultiple");
  using SPSSig = shared::SPSError(shared::SPSExecutorAddr,
                                  shared::SPSSequence<shared::SPSExecutorAddr>);
};

} // namespace llvm::orc::rt::sps_ci

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_SIMPLENATIVEMEMORYMAPSPSCI_H
