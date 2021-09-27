//===---- EPCGenericRTDyldMemoryManager.h - EPC-based MemMgr ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a RuntimeDyld::MemoryManager that uses EPC and the ORC runtime
// bootstrap functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICRTDYLDMEMORYMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICRTDYLDMEMORYMANAGER_H

#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

/// Remote-mapped RuntimeDyld-compatible memory manager.
class EPCGenericRTDyldMemoryManager : public RuntimeDyld::MemoryManager {
public:
  /// Symbol addresses for memory access.
  struct SymbolAddrs {
    ExecutorAddr Instance;
    ExecutorAddr Reserve;
    ExecutorAddr Finalize;
    ExecutorAddr Deallocate;
    ExecutorAddr RegisterEHFrame;
    ExecutorAddr DeregisterEHFrame;
  };

  /// Create an EPCGenericRTDyldMemoryManager using the given EPC, looking up
  /// the default symbol names in the bootstrap symbol set.
  static Expected<std::unique_ptr<EPCGenericRTDyldMemoryManager>>
  CreateWithDefaultBootstrapSymbols(ExecutorProcessControl &EPC);

  /// Create an EPCGenericRTDyldMemoryManager using the given EPC and symbol
  /// addrs.
  EPCGenericRTDyldMemoryManager(ExecutorProcessControl &EPC, SymbolAddrs SAs);

  EPCGenericRTDyldMemoryManager(const EPCGenericRTDyldMemoryManager &) = delete;
  EPCGenericRTDyldMemoryManager &
  operator=(const EPCGenericRTDyldMemoryManager &) = delete;
  EPCGenericRTDyldMemoryManager(EPCGenericRTDyldMemoryManager &&) = delete;
  EPCGenericRTDyldMemoryManager &
  operator=(EPCGenericRTDyldMemoryManager &&) = delete;
  ~EPCGenericRTDyldMemoryManager();

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override;

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool IsReadOnly) override;

  void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                              uintptr_t RODataSize, uint32_t RODataAlign,
                              uintptr_t RWDataSize,
                              uint32_t RWDataAlign) override;

  bool needsToReserveAllocationSpace() override;

  void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr, size_t Size) override;

  void deregisterEHFrames() override;

  void notifyObjectLoaded(RuntimeDyld &Dyld,
                          const object::ObjectFile &Obj) override;

  bool finalizeMemory(std::string *ErrMsg = nullptr) override;

private:
  struct Alloc {
  public:
    Alloc(uint64_t Size, unsigned Align)
        : Size(Size), Align(Align),
          Contents(std::make_unique<uint8_t[]>(Size + Align - 1)) {}

    uint64_t Size;
    unsigned Align;
    std::unique_ptr<uint8_t[]> Contents;
    ExecutorAddr RemoteAddr;
  };

  struct EHFrame {
    ExecutorAddr Addr;
    uint64_t Size;
  };

  // Group of section allocations to be allocated together in the executor. The
  // RemoteCodeAddr will stand in as the id of the group for deallocation
  // purposes.
  struct AllocGroup {
    AllocGroup() = default;
    AllocGroup(const AllocGroup &) = delete;
    AllocGroup &operator=(const AllocGroup &) = delete;
    AllocGroup(AllocGroup &&) = default;
    AllocGroup &operator=(AllocGroup &&) = default;

    ExecutorAddrRange RemoteCode;
    ExecutorAddrRange RemoteROData;
    ExecutorAddrRange RemoteRWData;
    std::vector<EHFrame> UnfinalizedEHFrames;
    std::vector<Alloc> CodeAllocs, RODataAllocs, RWDataAllocs;
  };

  // Maps all allocations in Allocs to aligned blocks
  void mapAllocsToRemoteAddrs(RuntimeDyld &Dyld, std::vector<Alloc> &Allocs,
                              ExecutorAddr NextAddr);

  ExecutorProcessControl &EPC;
  SymbolAddrs SAs;

  std::mutex M;
  std::vector<AllocGroup> Unmapped;
  std::vector<AllocGroup> Unfinalized;
  std::vector<ExecutorAddr> FinalizedAllocs;
  std::string ErrMsg;
};

} // end namespace orc
} // end namespace llvm

#undef DEBUG_TYPE

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICRTDYLDMEMORYMANAGER_H
