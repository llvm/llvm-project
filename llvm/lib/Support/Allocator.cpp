//===--- Allocator.cpp - Simple memory allocation abstraction -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BumpPtrAllocator interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Allocator.h"
#include "llvm/Support/Recycler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Streams.h"
#include <ostream>
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
  void *Allocate(size_t AllocSize, size_t Alignment, MemRegion **RegPtr) {
    
    char* Result = (char*) (((uintptr_t) (NextPtr+Alignment-1)) 
                            & ~((uintptr_t) Alignment-1));

    // Speculate the new value of NextPtr.
    char* NextPtrTmp = Result + AllocSize;
    
    // If we are still within the current region, return Result.
    if (unsigned (NextPtrTmp - (char*) this) <= RegionSize) {
      NextPtr = NextPtrTmp;
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
  
  /// Deallocate - Recursively release all memory for this and its next regions
  /// to the system.
  void Deallocate() {
    MemRegion *next = Next;
    free(this);
    if (next)
      next->Deallocate();
  }

  /// DeallocateAllButLast - Recursively release all memory for this and its
  /// next regions to the system stopping at the last region in the list.
  /// Returns the pointer to the last region.
  MemRegion *DeallocateAllButLast() {
    MemRegion *next = Next;
    if (!next)
      return this;
    free(this);
    return next->DeallocateAllButLast();
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

void BumpPtrAllocator::Reset() {
  MemRegion *MRP = (MemRegion*)TheMemory;
  MRP = MRP->DeallocateAllButLast();
  MRP->Init(4096, 1, 0);
  TheMemory = MRP;
}

void *BumpPtrAllocator::Allocate(size_t Size, size_t Align) {
  MemRegion *MRP = (MemRegion*)TheMemory;
  void *Ptr = MRP->Allocate(Size, Align, &MRP);
  TheMemory = MRP;
  return Ptr;
}

void BumpPtrAllocator::PrintStats() const {
  unsigned BytesUsed = 0;
  unsigned NumRegions = 0;
  const MemRegion *R = (MemRegion*)TheMemory;
  for (; R; R = R->getNext(), ++NumRegions)
    BytesUsed += R->getNumBytesAllocated();

  cerr << "\nNumber of memory regions: " << NumRegions << "\n";
  cerr << "Bytes allocated: " << BytesUsed << "\n";
}

void llvm::PrintRecyclerStats(size_t Size,
                              size_t Align,
                              size_t FreeListSize) {
  cerr << "Recycler element size: " << Size << '\n';
  cerr << "Recycler element alignment: " << Align << '\n';
  cerr << "Number of elements free for recycling: " << FreeListSize << '\n';
}
