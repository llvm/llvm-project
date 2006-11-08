//===--- Allocator.cpp - Simple memory allocation abstraction -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BumpPtrAllocator interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Allocator.h"
#include "llvm/Support/DataTypes.h"
#include <iostream>
using namespace llvm;

//===----------------------------------------------------------------------===//
// MemRegion class implementation
//===----------------------------------------------------------------------===//

namespace {
/// MemRegion - This is one chunk of the BumpPtrAllocator.
class MemRegion {
  unsigned RegionSize;
  MemRegion *Next;
  char *NextPtr;
public:
  void Init(unsigned size, unsigned Alignment, MemRegion *next) {
    RegionSize = size;
    Next = next;
    NextPtr = (char*)(this+1);
    
    // Align NextPtr.
    NextPtr = (char*)((intptr_t)(NextPtr+Alignment-1) &
                      ~(intptr_t)(Alignment-1));
  }
  
  const MemRegion *getNext() const { return Next; }
  unsigned getNumBytesAllocated() const {
    return NextPtr-(const char*)this;
  }
  
  /// Allocate - Allocate and return at least the specified number of bytes.
  ///
  void *Allocate(unsigned AllocSize, unsigned Alignment, MemRegion **RegPtr) {
    // Round size up to an even multiple of the alignment.
    AllocSize = (AllocSize+Alignment-1) & ~(Alignment-1);
    
    // If there is space in this region, return it.
    if (unsigned(NextPtr+AllocSize-(char*)this) <= RegionSize) {
      void *Result = NextPtr;
      NextPtr += AllocSize;
      return Result;
    }
    
    // Otherwise, we have to allocate a new chunk.  Create one twice as big as
    // this one.
    MemRegion *NewRegion = (MemRegion *)malloc(RegionSize*2);
    NewRegion->Init(RegionSize*2, Alignment, this);

    // Update the current "first region" pointer  to point to the new region.
    *RegPtr = NewRegion;
    
    // Try allocating from it now.
    return NewRegion->Allocate(AllocSize, Alignment, RegPtr);
  }
  
  /// Deallocate - Release all memory for this region to the system.
  ///
  void Deallocate() {
    MemRegion *next = Next;
    free(this);
    if (next)
      next->Deallocate();
  }
};
}

//===----------------------------------------------------------------------===//
// BumpPtrAllocator class implementation
//===----------------------------------------------------------------------===//

BumpPtrAllocator::BumpPtrAllocator() {
  TheMemory = malloc(4096);
  ((MemRegion*)TheMemory)->Init(4096, 1, 0);
}

BumpPtrAllocator::~BumpPtrAllocator() {
  ((MemRegion*)TheMemory)->Deallocate();
}

void *BumpPtrAllocator::Allocate(unsigned Size, unsigned Align) {
  return ((MemRegion*)TheMemory)->Allocate(Size, Align,(MemRegion**)&TheMemory);
}

void BumpPtrAllocator::PrintStats() const {
  unsigned BytesUsed = 0;
  unsigned NumRegions = 0;
  const MemRegion *R = (MemRegion*)TheMemory;
  for (; R; R = R->getNext(), ++NumRegions)
    BytesUsed += R->getNumBytesAllocated();

  std::cerr << "\nNumber of memory regions: " << NumRegions << "\n";
  std::cerr << "Bytes allocated: " << BytesUsed << "\n";
}
