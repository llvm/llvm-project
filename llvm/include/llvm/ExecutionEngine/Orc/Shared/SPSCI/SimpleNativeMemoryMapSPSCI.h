//===-- SimpleNativeMemoryMapSPSCI.h - SPS CI for mem mgmt ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS controller-interface descriptors for the runtime's SimpleNativeMemoryMap
// interface. The instance address of the executor-side manager is passed as
// the first argument to each call.
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
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>

namespace llvm::orc::rt::sps_ci {

/// The executor-side memory-manager instance. This is a data symbol (the
/// allocator object) -- passed as the first argument to each call below -- not
/// a wrapper to call.
inline constexpr char SimpleNativeMemoryMapInstanceName[] =
    "orc_rt_ci_SimpleNativeMemoryMap_Instance";

/// Reserve an address range of the given size; returns its base.
struct MemMgrReserve {
  static constexpr char Name[] = "orc_rt_ci_sps_SimpleNativeMemoryMap_reserve";
  using SPSSig = shared::SPSExpected<shared::SPSExecutorAddr>(
      shared::SPSExecutorAddr, uint64_t);
};

/// Apply a finalize request; returns a key for the initialized allocation.
struct MemMgrInitialize {
  static constexpr char Name[] =
      "orc_rt_ci_sps_SimpleNativeMemoryMap_initialize";
  using SPSSig = shared::SPSExpected<shared::SPSExecutorAddr>(
      shared::SPSExecutorAddr, shared::SPSFinalizeRequest);
};

/// Deinitialize the allocations with the given base addresses (running their
/// deallocation actions) without releasing their memory.
struct MemMgrDeinitialize {
  static constexpr char Name[] =
      "orc_rt_ci_sps_SimpleNativeMemoryMap_deinitializeMultiple";
  using SPSSig = shared::SPSError(shared::SPSExecutorAddr,
                                  shared::SPSSequence<shared::SPSExecutorAddr>);
};

/// Release the allocations with the given base addresses.
struct MemMgrRelease {
  static constexpr char Name[] =
      "orc_rt_ci_sps_SimpleNativeMemoryMap_releaseMultiple";
  using SPSSig = shared::SPSError(shared::SPSExecutorAddr,
                                  shared::SPSSequence<shared::SPSExecutorAddr>);
};

} // namespace llvm::orc::rt::sps_ci

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_SIMPLENATIVEMEMORYMAPSPSCI_H
