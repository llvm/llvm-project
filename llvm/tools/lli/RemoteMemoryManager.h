//===- RemoteMemoryManager.h - LLI MCJIT recording memory manager ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This memory manager allocates local storage and keeps a record of each
// allocation. Iterators are provided for all data and code allocations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLI_REMOTEMEMORYMANAGER_H
#define LLVM_TOOLS_LLI_REMOTEMEMORYMANAGER_H

#include "RemoteTarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Memory.h"
#include <utility>

namespace llvm {

class RemoteMemoryManager : public RTDyldMemoryManager {
public:
  // Notice that this structure takes ownership of the memory allocated.
  struct Allocation {
    Allocation() {}
    Allocation(sys::MemoryBlock mb, unsigned a, bool code)
      : MB(mb), Alignment(a), IsCode(code) {}

    sys::MemoryBlock  MB;
    unsigned          Alignment;
    bool              IsCode;
  };

private:
  // This vector contains Allocation objects for all sections which we have
  // allocated.  This vector effectively owns the memory associated with the
  // allocations.
  SmallVector<Allocation, 2>  AllocatedSections;

  // This vector contains pointers to Allocation objects for any sections we
  // have allocated locally but have not yet remapped for the remote target.
  // When we receive notification of a completed module load, we will map
  // these sections into the remote target.
  SmallVector<Allocation, 2>  UnmappedSections;

  // This map tracks the sections we have remapped for the remote target
  // but have not yet copied to the target.
  DenseMap<uint64_t, Allocation>  MappedSections;

  // FIXME: This is part of a work around to keep sections near one another
  // when MCJIT performs relocations after code emission but before
  // the generated code is moved to the remote target.
  sys::MemoryBlock Near;
  sys::MemoryBlock allocateSection(uintptr_t Size);

  RemoteTarget *Target;

public:
  RemoteMemoryManager() : Target(nullptr) {}
  ~RemoteMemoryManager() override;

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override;

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool IsReadOnly) override;

  // For now, remote symbol resolution is not support in lli.  The MCJIT
  // interface does support this, but clients must provide their own
  // mechanism for finding remote symbol addresses.  MCJIT will resolve
  // symbols from Modules it contains.
  uint64_t getSymbolAddress(const std::string &Name) override { return 0; }

  void notifyObjectLoaded(ExecutionEngine *EE,
                          const object::ObjectFile &Obj) override;

  bool finalizeMemory(std::string *ErrMsg) override;

  // For now, remote EH frame registration isn't supported.  Remote symbol
  // resolution is a prerequisite to supporting remote EH frame registration.
  void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                        size_t Size) override {}
  void deregisterEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                          size_t Size) override {}

  // This is a non-interface function used by lli
  void setRemoteTarget(RemoteTarget *T) { Target = T; }
};

} // end namespace llvm

#endif
