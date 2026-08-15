//===- EPCGenericJITLinkMemoryManager.h - EPC-based mem manager -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements JITLinkMemoryManager by calling executor-side wrapper functions
// through Proxy objects.
//
// This simplifies the implementaton of new ExecutorProcessControl instances,
// as this implementation will always work (at the cost of some performance
// overhead for the calls).
//
// This header is protocol-agnostic. To build an instance that targets the ORC
// runtime's SPS controller interface, see EPCGenericJITLinkMemoryManagerSPS.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGER_H

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Proxy.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/Support/Compiler.h"

#include <cstdint>

namespace llvm {
namespace orc {

class LLVM_ABI EPCGenericJITLinkMemoryManager
    : public jitlink::JITLinkMemoryManager {
public:
  /// Reserve an address range of the given size; returns its base.
  using ReserveProxy = Proxy<Expected<ExecutorAddr>(ExecutorAddr, uint64_t)>;

  /// Apply a finalize request; returns a key for the initialized allocation.
  using InitializeProxy =
      Proxy<Expected<ExecutorAddr>(ExecutorAddr, tpctypes::FinalizeRequest)>;

  /// Deinitialize the allocations with the given base addresses (running their
  /// deallocation actions) without releasing their memory.
  using DeinitializeProxy = Proxy<Error(ExecutorAddr, ArrayRef<ExecutorAddr>)>;

  /// Release the allocations with the given base addresses.
  using ReleaseProxy = Proxy<Error(ExecutorAddr, ArrayRef<ExecutorAddr>)>;

  /// The resolved controller-side handle to an executor-side memory manager:
  /// the address of the allocator instance (passed as the first argument to
  /// each call) plus the proxies for its functions. These are
  /// protocol-agnostic: sps::createEPCGenericJITLinkMemoryManager populates
  /// them for the runtime's SPS controller interface, but a client targeting a
  /// different protocol can build its own Bindings and pass them to the
  /// constructor.
  ///
  /// Deinitialize is part of the interface but is not currently used by this
  /// manager.
  struct Bindings {
    ExecutorAddr Instance;
    ReserveProxy Reserve;
    InitializeProxy Initialize;
    DeinitializeProxy Deinitialize;
    ReleaseProxy Release;
  };

  /// Create an EPCGenericJITLinkMemoryManager instance from a given set of
  /// memory-manager bindings.
  EPCGenericJITLinkMemoryManager(ExecutionSession &ES, Bindings B)
      : ES(ES), B(std::move(B)) {}

  void allocate(const jitlink::JITLinkDylib *JD, jitlink::LinkGraph &G,
                OnAllocatedFunction OnAllocated) override;

  // Use overloads from base class.
  using JITLinkMemoryManager::allocate;

  void deallocate(std::vector<FinalizedAlloc> Allocs,
                  OnDeallocatedFunction OnDeallocated) override;

  // Use overloads from base class.
  using JITLinkMemoryManager::deallocate;

private:
  class InFlightAlloc;

  void completeAllocation(ExecutorAddr AllocAddr, jitlink::BasicLayout BL,
                          OnAllocatedFunction OnAllocated);

  ExecutionSession &ES;
  Bindings B;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGER_H
