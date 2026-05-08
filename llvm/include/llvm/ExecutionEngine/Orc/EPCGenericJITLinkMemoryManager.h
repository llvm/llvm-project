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
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace orc {

class LLVM_ABI EPCGenericJITLinkMemoryManager
    : public jitlink::JITLinkMemoryManager {
public:
  /// Symbol addresses for memory management implementation.
  struct SymbolAddrs {
    ExecutorAddr Allocator;
    ExecutorAddr Reserve;
    ExecutorAddr Initialize;
    ExecutorAddr Deinitialize;
    ExecutorAddr Release;
  };

  /// Symbol names for memory management implementation.
  struct SymbolNames {
    StringRef AllocatorName;
    StringRef ReserveName;
    StringRef InitializeName;
    StringRef DeinitializeName;
    StringRef ReleaseName;
  };

  /// Default symbol names for the ORC runtime's SimpleNativeMemoryMap SPS
  /// interface.
  static const SymbolNames orc_rt_SimpleNativeMemoryMapSPSSymbols;

  /// Create an EPCGenericJITLinkMemoryManager instance from a given set of
  /// function addrs.
  EPCGenericJITLinkMemoryManager(ExecutorProcessControl &EPC, SymbolAddrs SAs)
      : EPC(EPC), SAs(SAs) {}

  /// Create an EPCGenericJITLinkMemoryManager using the given implementation
  /// symbol names. These will be looked up in the given JITDylib.
  static Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
  Create(JITDylib &JD,
         SymbolNames SNs = orc_rt_SimpleNativeMemoryMapSPSSymbols);

  /// Create an EPCGenericJITLinkMemoryManager using the given implementation
  /// symbol names. These will be looked up in the given ExecutionSession's
  /// Bootstrap JITDylib.
  static Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
  Create(ExecutionSession &ES,
         SymbolNames SNs = orc_rt_SimpleNativeMemoryMapSPSSymbols);

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

  ExecutorProcessControl &EPC;
  SymbolAddrs SAs;
};

namespace shared {

/// FIXME: This specialization should be moved into TargetProcessControlTypes.h
///        (or wherever those types get merged to) once ORC depends on JITLink.
template <>
class SPSSerializationTraits<SPSExecutorAddr,
                             jitlink::JITLinkMemoryManager::FinalizedAlloc> {
public:
  static size_t size(const jitlink::JITLinkMemoryManager::FinalizedAlloc &FA) {
    return SPSArgList<SPSExecutorAddr>::size(ExecutorAddr(FA.getAddress()));
  }

  static bool
  serialize(SPSOutputBuffer &OB,
            const jitlink::JITLinkMemoryManager::FinalizedAlloc &FA) {
    return SPSArgList<SPSExecutorAddr>::serialize(
        OB, ExecutorAddr(FA.getAddress()));
  }

  static bool deserialize(SPSInputBuffer &IB,
                          jitlink::JITLinkMemoryManager::FinalizedAlloc &FA) {
    ExecutorAddr A;
    if (!SPSArgList<SPSExecutorAddr>::deserialize(IB, A))
      return false;
    FA = jitlink::JITLinkMemoryManager::FinalizedAlloc(A);
    return true;
  }
};

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICJITLINKMEMORYMANAGER_H
