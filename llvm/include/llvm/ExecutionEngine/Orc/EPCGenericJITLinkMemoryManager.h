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
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/ExecutionEngine/Orc/SimpleMemoryMap.h"
#include "llvm/Support/Compiler.h"

#include <cstdint>

namespace llvm {
namespace orc {

class LLVM_ABI EPCGenericJITLinkMemoryManager
    : public jitlink::JITLinkMemoryManager {
public:
  /// Create an EPCGenericJITLinkMemoryManager instance from a given set of
  /// memory-manager bindings.
  EPCGenericJITLinkMemoryManager(ExecutionSession &ES,
                                 SimpleMemoryMapBindings B)
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
  SimpleMemoryMapBindings B;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGER_H
