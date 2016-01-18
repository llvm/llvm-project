//===---- RemoteMemoryManager.cpp - Recording memory manager --------------===//
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

#include "RemoteMemoryManager.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "lli"

RemoteMemoryManager::~RemoteMemoryManager() {
  for (SmallVector<Allocation, 2>::iterator
         I = AllocatedSections.begin(), E = AllocatedSections.end();
       I != E; ++I)
    sys::Memory::releaseMappedMemory(I->MB);
}

uint8_t *RemoteMemoryManager::
allocateCodeSection(uintptr_t Size, unsigned Alignment, unsigned SectionID,
                    StringRef SectionName) {
  // The recording memory manager is just a local copy of the remote target.
  // The alignment requirement is just stored here for later use. Regular
  // heap storage is sufficient here, but we're using mapped memory to work
  // around a bug in MCJIT.
  sys::MemoryBlock Block = allocateSection(Size);
  // AllocatedSections will own this memory.
  AllocatedSections.push_back( Allocation(Block, Alignment, true) );
  // UnmappedSections has the same information but does not own the memory.
  UnmappedSections.push_back( Allocation(Block, Alignment, true) );
  return (uint8_t*)Block.base();
}

uint8_t *RemoteMemoryManager::
allocateDataSection(uintptr_t Size, unsigned Alignment,
                    unsigned SectionID, StringRef SectionName,
                    bool IsReadOnly) {
  // The recording memory manager is just a local copy of the remote target.
  // The alignment requirement is just stored here for later use. Regular
  // heap storage is sufficient here, but we're using mapped memory to work
  // around a bug in MCJIT.
  sys::MemoryBlock Block = allocateSection(Size);
  // AllocatedSections will own this memory.
  AllocatedSections.push_back( Allocation(Block, Alignment, false) );
  // UnmappedSections has the same information but does not own the memory.
  UnmappedSections.push_back( Allocation(Block, Alignment, false) );
  return (uint8_t*)Block.base();
}

sys::MemoryBlock RemoteMemoryManager::allocateSection(uintptr_t Size) {
  std::error_code ec;
  sys::MemoryBlock MB = sys::Memory::allocateMappedMemory(Size,
                                                          &Near,
                                                          sys::Memory::MF_READ |
                                                          sys::Memory::MF_WRITE,
                                                          ec);
  assert(!ec && MB.base());

  // FIXME: This is part of a work around to keep sections near one another
  // when MCJIT performs relocations after code emission but before
  // the generated code is moved to the remote target.
  // Save this address as the basis for our next request
  Near = MB;
  return MB;
}

void RemoteMemoryManager::notifyObjectLoaded(ExecutionEngine *EE,
                                             const object::ObjectFile &Obj) {
  // The client should have called setRemoteTarget() before triggering any
  // code generation.
  assert(Target);
  if (!Target)
    return;

  // FIXME: Make this function thread safe.

  // Lay out our sections in order, with all the code sections first, then
  // all the data sections.
  uint64_t CurOffset = 0;
  unsigned MaxAlign = Target->getPageAlignment();
  SmallVector<std::pair<Allocation, uint64_t>, 16> Offsets;
  unsigned NumSections = UnmappedSections.size();
  // We're going to go through the list twice to separate code and data, but
  // it's a very small list, so that's OK.
  for (size_t i = 0, e = NumSections; i != e; ++i) {
    Allocation &Section = UnmappedSections[i];
    if (Section.IsCode) {
      unsigned Size = Section.MB.size();
      unsigned Align = Section.Alignment;
      DEBUG(dbgs() << "code region: size " << Size
                  << ", alignment " << Align << "\n");
      // Align the current offset up to whatever is needed for the next
      // section.
      CurOffset = (CurOffset + Align - 1) / Align * Align;
      // Save off the address of the new section and allocate its space.
      Offsets.push_back(std::pair<Allocation,uint64_t>(Section, CurOffset));
      CurOffset += Size;
    }
  }
  // Adjust to keep code and data aligned on separate pages.
  CurOffset = (CurOffset + MaxAlign - 1) / MaxAlign * MaxAlign;
  for (size_t i = 0, e = NumSections; i != e; ++i) {
    Allocation &Section = UnmappedSections[i];
    if (!Section.IsCode) {
      unsigned Size = Section.MB.size();
      unsigned Align = Section.Alignment;
      DEBUG(dbgs() << "data region: size " << Size
                  << ", alignment " << Align << "\n");
      // Align the current offset up to whatever is needed for the next
      // section.
      CurOffset = (CurOffset + Align - 1) / Align * Align;
      // Save off the address of the new section and allocate its space.
      Offsets.push_back(std::pair<Allocation,uint64_t>(Section, CurOffset));
      CurOffset += Size;
    }
  }

  // Allocate space in the remote target.
  uint64_t RemoteAddr;
  if (!Target->allocateSpace(CurOffset, MaxAlign, RemoteAddr))
    report_fatal_error(Target->getErrorMsg());

  // Map the section addresses so relocations will get updated in the local
  // copies of the sections.
  for (unsigned i = 0, e = Offsets.size(); i != e; ++i) {
    uint64_t Addr = RemoteAddr + Offsets[i].second;
    EE->mapSectionAddress(const_cast<void*>(Offsets[i].first.MB.base()), Addr);

    DEBUG(dbgs() << "  Mapping local: " << Offsets[i].first.MB.base()
                 << " to remote: 0x" << format("%llx", Addr) << "\n");

    MappedSections[Addr] = Offsets[i].first;
  }

  UnmappedSections.clear();
}

bool RemoteMemoryManager::finalizeMemory(std::string *ErrMsg) {
  // FIXME: Make this function thread safe.
  for (DenseMap<uint64_t, Allocation>::iterator
         I = MappedSections.begin(), E = MappedSections.end();
       I != E; ++I) {
    uint64_t RemoteAddr = I->first;
    const Allocation &Section = I->second;
    if (Section.IsCode) {
      if (!Target->loadCode(RemoteAddr, Section.MB.base(), Section.MB.size()))
        report_fatal_error(Target->getErrorMsg());
      DEBUG(dbgs() << "  loading code: " << Section.MB.base()
            << " to remote: 0x" << format("%llx", RemoteAddr) << "\n");
    } else {
      if (!Target->loadData(RemoteAddr, Section.MB.base(), Section.MB.size()))
        report_fatal_error(Target->getErrorMsg());
      DEBUG(dbgs() << "  loading data: " << Section.MB.base()
            << " to remote: 0x" << format("%llx", RemoteAddr) << "\n");
    }
  }

  MappedSections.clear();

  return false;
}
