//===- EPCGenericJITLinkMemoryManager.h - EPC-based mem manager -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements JITLinkMemoryManager by making remove calls via
// ExecutorProcessControl::callWrapperAsync.
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

namespace llvm {
namespace orc {

class EPCGenericJITLinkMemoryManager : public jitlink::JITLinkMemoryManager {
public:
  /// Function addresses for memory access.
  struct SymbolAddrs {
    ExecutorAddr Allocator;
    ExecutorAddr Reserve;
    ExecutorAddr Finalize;
    ExecutorAddr Deallocate;
  };

  /// Create an EPCGenericJITLinkMemoryManager instance from a given set of
  /// function addrs.
  EPCGenericJITLinkMemoryManager(ExecutorProcessControl &EPC, SymbolAddrs SAs)
      : EPC(EPC), SAs(SAs) {}

  Expected<std::unique_ptr<Allocation>>
  allocate(const jitlink::JITLinkDylib *JD,
           const SegmentsRequestMap &Request) override;

private:
  class Alloc;

  ExecutorProcessControl &EPC;
  SymbolAddrs SAs;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGER_H
