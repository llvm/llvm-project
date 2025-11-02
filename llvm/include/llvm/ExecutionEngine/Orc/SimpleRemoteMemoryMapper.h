//===- SimpleRemoteMemoryMapper.h - Remote memory mapper --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A simple memory mapper that uses EPC calls to implement reserve, initialize,
// deinitialize, and release.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SIMPLEREMOTEMEMORYMAPPER_H
#define LLVM_EXECUTIONENGINE_ORC_SIMPLEREMOTEMEMORYMAPPER_H

#include "llvm/ExecutionEngine/Orc/MemoryMapper.h"

namespace llvm::orc {

/// Manages remote memory by making SPS-based EPC calls.
class LLVM_ABI SimpleRemoteMemoryMapper final : public MemoryMapper {
public:
  struct SymbolAddrs {
    ExecutorAddr Instance;
    ExecutorAddr Reserve;
    ExecutorAddr Initialize;
    ExecutorAddr Deinitialize;
    ExecutorAddr Release;
  };

  SimpleRemoteMemoryMapper(ExecutorProcessControl &EPC, SymbolAddrs SAs);

  static Expected<std::unique_ptr<SimpleRemoteMemoryMapper>>
  Create(ExecutorProcessControl &EPC, SymbolAddrs SAs) {
    return std::make_unique<SimpleRemoteMemoryMapper>(EPC, SAs);
  }

  unsigned int getPageSize() override { return EPC.getPageSize(); }

  /// Reserves memory in the remote process by calling a remote
  /// SPS-wrapper-function with signature
  ///
  ///   SPSExpected<SPSExecutorAddr>(uint64_t Size).
  ///
  /// On success, returns the base address of the reserved range.
  void reserve(size_t NumBytes, OnReservedFunction OnReserved) override;

  char *prepare(jitlink::LinkGraph &G, ExecutorAddr Addr,
                size_t ContentSize) override;

  /// Initializes memory within a previously reserved region (applying
  /// protections and running any finalization actions) by calling a remote
  /// SPS-wrapper-function with signature
  ///
  ///   SPSExpected<SPSExecutorAddr>(SPSFinalizeRequest)
  ///
  /// On success, returns a key that can be used to deinitialize the region.
  void initialize(AllocInfo &AI, OnInitializedFunction OnInitialized) override;

  /// Given a series of keys from previous initialize calls, deinitialize
  /// previously initialized memory regions (running dealloc actions, resetting
  /// permissions and decommitting if possible) by calling a remote
  /// SPS-wrapper-function with signature
  ///
  ///   SPSError(SPSSequence<SPSExecutorAddr> Keys)
  ///
  void deinitialize(ArrayRef<ExecutorAddr> Allocations,
                    OnDeinitializedFunction OnDeInitialized) override;

  /// Given a sequence of base addresses from previous reserve calls, release
  /// the underlying ranges (deinitializing any remaining regions within them)
  /// by calling a remote SPS-wrapper-function with signature
  ///
  ///   SPSError(SPSSequence<SPSExecutorAddr> Bases)
  ///
  void release(ArrayRef<ExecutorAddr> Reservations,
               OnReleasedFunction OnRelease) override;

private:
  ExecutorProcessControl &EPC;
  SymbolAddrs SAs;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_SIMPLEREMOTEMEMORYMAPPER_H
