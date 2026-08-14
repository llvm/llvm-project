//===- EPCGenericJITLinkMemoryManager.h - EPC-based mem manager -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements JITLinkMemoryManager by calling executor-side wrapper functions
// through rt::Proxy objects.
//
// This simplifies the implementaton of new ExecutorProcessControl instances,
// as this implementation will always work (at the cost of some performance
// overhead for the calls).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGER_H

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/GenericMemoryManagerProxies.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace orc {

class LLVM_ABI EPCGenericJITLinkMemoryManager
    : public jitlink::JITLinkMemoryManager {
public:
  /// The resolved controller-side handle to an executor-side memory manager:
  /// the address of the allocator instance (passed as the first argument to
  /// each call) plus the proxies for its functions. These are
  /// protocol-agnostic: the Create methods populate them for the runtime's SPS
  /// controller interface, but a client targeting a different protocol can
  /// build its own Bindings and pass them to the constructor.
  ///
  /// Deinitialize is part of the interface but is not currently used by this
  /// manager.
  struct Bindings {
    ExecutorAddr Instance;
    rt::MemMgrReserveProxy Reserve;
    rt::MemMgrInitializeProxy Initialize;
    rt::MemMgrDeinitializeProxy Deinitialize;
    rt::MemMgrReleaseProxy Release;
  };

  /// Create an EPCGenericJITLinkMemoryManager instance from a given set of
  /// memory-manager bindings.
  EPCGenericJITLinkMemoryManager(ExecutionSession &ES, Bindings B)
      : ES(ES), B(std::move(B)) {}

  /// Create an EPCGenericJITLinkMemoryManager for the ORC runtime's
  /// SimpleNativeMemoryMap interface, resolving its symbols in the given
  /// JITDylib.
  static Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
  Create(JITDylib &JD);

  /// Create an EPCGenericJITLinkMemoryManager for the ORC runtime's
  /// SimpleNativeMemoryMap interface, resolving its symbols in the given
  /// ExecutionSession's bootstrap JITDylib.
  static Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
  Create(ExecutionSession &ES);

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
