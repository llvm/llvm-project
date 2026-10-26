//===- SimpleRemoteMemoryMapper.h - Remote memory mapper --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A simple memory mapper that implements reserve, initialize, deinitialize and
// release by driving an executor-side memory manager through Proxy objects.
//
// This header is protocol-agnostic. To build the bindings for the ORC runtime's
// SPS controller interface, see SimpleMemoryMapSPS.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SIMPLEREMOTEMEMORYMAPPER_H
#define LLVM_EXECUTIONENGINE_ORC_SIMPLEREMOTEMEMORYMAPPER_H

#include "llvm/ExecutionEngine/Orc/MemoryMapper.h"
#include "llvm/ExecutionEngine/Orc/SimpleMemoryMap.h"

namespace llvm::orc {

/// Manages remote memory by driving an executor-side memory manager.
class LLVM_ABI SimpleRemoteMemoryMapper final : public MemoryMapper {
public:
  /// Create a SimpleRemoteMemoryMapper from a given set of memory-manager
  /// bindings.
  SimpleRemoteMemoryMapper(ExecutionSession &ES, SimpleMemoryMapBindings B);

  static Expected<std::unique_ptr<SimpleRemoteMemoryMapper>>
  Create(ExecutionSession &ES, SimpleMemoryMapBindings B) {
    return std::make_unique<SimpleRemoteMemoryMapper>(ES, std::move(B));
  }

  unsigned int getPageSize() override {
    return ES.getExecutorProcessControl().getPageSize();
  }

  /// Reserves memory in the executor, returning the base address of the
  /// reserved range on success.
  void reserve(size_t NumBytes, OnReservedFunction OnReserved) override;

  char *prepare(jitlink::LinkGraph &G, ExecutorAddr Addr,
                size_t ContentSize) override;

  /// Initializes memory within a previously reserved region, applying
  /// protections and running any finalization actions.
  ///
  /// On success, returns a key that can be used to deinitialize the region.
  void initialize(AllocInfo &AI, OnInitializedFunction OnInitialized) override;

  /// Given a series of keys from previous initialize calls, deinitialize
  /// previously initialized memory regions: run their dealloc actions, reset
  /// permissions, and decommit if possible.
  void deinitialize(ArrayRef<ExecutorAddr> Allocations,
                    OnDeinitializedFunction OnDeInitialized) override;

  /// Given a sequence of base addresses from previous reserve calls, release
  /// the underlying ranges, deinitializing any remaining regions within them.
  void release(ArrayRef<ExecutorAddr> Reservations,
               OnReleasedFunction OnRelease) override;

private:
  ExecutionSession &ES;
  SimpleMemoryMapBindings B;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_SIMPLEREMOTEMEMORYMAPPER_H
