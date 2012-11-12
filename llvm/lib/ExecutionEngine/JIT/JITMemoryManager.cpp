//===-- JITMemoryManager.cpp - Memory Allocator for JIT'd code ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DefaultJITMemoryManager class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/GlobalValue.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Config/config.h"
#include <vector>
#include <cassert>
#include <climits>
#include <cstring>

#if defined(__linux__)
#if defined(HAVE_SYS_STAT_H)
#include <sys/stat.h>
#endif
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace llvm;

STATISTIC(NumSlabs, "Number of slabs of memory allocated by the JIT");

JITMemoryManager::~JITMemoryManager() {}

//===----------------------------------------------------------------------===//
// Memory Block Implementation.
//===----------------------------------------------------------------------===//

namespace {
  /// MemoryRangeHeader - For a range of memory, this is the header that we put
  /// on the block of memory.  It is carefully crafted to be one word of memory.
  /// Allocated blocks have just this header, free'd blocks have FreeRangeHeader
  /// which starts with this.
  struct FreeRangeHeader;
  struct MemoryRangeHeader {
    /// ThisAllocated - This is true if this block is currently allocated.  If
    /// not, this can be converted to a FreeRangeHeader.
    unsigned ThisAllocated : 1;

    /// PrevAllocated - Keep track of whether the block immediately before us is
    /// allocated.  If not, the word immediately before this header is the size
    /// of the previous block.
    unsigned PrevAllocated : 1;

    /// BlockSize - This is the size in bytes of this memory block,
    /// including this header.
    uintptr_t BlockSize : (sizeof(intptr_t)*CHAR_BIT - 2);


    /// getBlockAfter - Return the memory block immediately after this one.
    ///
    MemoryRangeHeader &getBlockAfter() const {
      return *(MemoryRangeHeader*)((char*)this+BlockSize);
    }

    /// getFreeBlockBefore - If the block before this one is free, return it,
    /// otherwise return null.
    FreeRangeHeader *getFreeBlockBefore() const {
      if (PrevAllocated) return 0;
      intptr_t PrevSize = ((intptr_t *)this)[-1];
      return (FreeRangeHeader*)((char*)this-PrevSize);
    }

    /// FreeBlock - Turn an allocated block into a free block, adjusting
    /// bits in the object headers, and adding an end of region memory block.
    FreeRangeHeader *FreeBlock(FreeRangeHeader *FreeList);

    /// TrimAllocationToSize - If this allocated block is significantly larger
    /// than NewSize, split it into two pieces (where the former is NewSize
    /// bytes, including the header), and add the new block to the free list.
    FreeRangeHeader *TrimAllocationToSize(FreeRangeHeader *FreeList,
                                          uint64_t NewSize);
  };

  /// FreeRangeHeader - For a memory block that isn't already allocated, this
  /// keeps track of the current block and has a pointer to the next free block.
  /// Free blocks are kept on a circularly linked list.
  struct FreeRangeHeader : public MemoryRangeHeader {
    FreeRangeHeader *Prev;
    FreeRangeHeader *Next;

    /// getMinBlockSize - Get the minimum size for a memory block.  Blocks
    /// smaller than this size cannot be created.
    static unsigned getMinBlockSize() {
      return sizeof(FreeRangeHeader)+sizeof(intptr_t);
    }

    /// SetEndOfBlockSizeMarker - The word at the end of every free block is
    /// known to be the size of the free block.  Set it for this block.
    void SetEndOfBlockSizeMarker() {
      void *EndOfBlock = (char*)this + BlockSize;
      ((intptr_t *)EndOfBlock)[-1] = BlockSize;
    }

    FreeRangeHeader *RemoveFromFreeList() {
      assert(Next->Prev == this && Prev->Next == this && "Freelist broken!");
      Next->Prev = Prev;
      return Prev->Next = Next;
    }

    void AddToFreeList(FreeRangeHeader *FreeList) {
      Next = FreeList;
      Prev = FreeList->Prev;
      Prev->Next = this;
      Next->Prev = this;
    }

    /// GrowBlock - The block after this block just got deallocated.  Merge it
    /// into the current block.
    void GrowBlock(uintptr_t NewSize);

    /// AllocateBlock - Mark this entire block allocated, updating freelists
    /// etc.  This returns a pointer to the circular free-list.
    FreeRangeHeader *AllocateBlock();
  };
}


/// AllocateBlock - Mark this entire block allocated, updating freelists
/// etc.  This returns a pointer to the circular free-list.
FreeRangeHeader *FreeRangeHeader::AllocateBlock() {
  assert(!ThisAllocated && !getBlockAfter().PrevAllocated &&
         "Cannot allocate an allocated block!");
  // Mark this block allocated.
  ThisAllocated = 1;
  getBlockAfter().PrevAllocated = 1;

  // Remove it from the free list.
  return RemoveFromFreeList();
}

/// FreeBlock - Turn an allocated block into a free block, adjusting
/// bits in the object headers, and adding an end of region memory block.
/// If possible, coalesce this block with neighboring blocks.  Return the
/// FreeRangeHeader to allocate from.
FreeRangeHeader *MemoryRangeHeader::FreeBlock(FreeRangeHeader *FreeList) {
  MemoryRangeHeader *FollowingBlock = &getBlockAfter();
  assert(ThisAllocated && "This block is already free!");
  assert(FollowingBlock->PrevAllocated && "Flags out of sync!");

  FreeRangeHeader *FreeListToReturn = FreeList;

  // If the block after this one is free, merge it into this block.
  if (!FollowingBlock->ThisAllocated) {
    FreeRangeHeader &FollowingFreeBlock = *(FreeRangeHeader *)FollowingBlock;
    // "FreeList" always needs to be a valid free block.  If we're about to
    // coalesce with it, update our notion of what the free list is.
    if (&FollowingFreeBlock == FreeList) {
      FreeList = FollowingFreeBlock.Next;
      FreeListToReturn = 0;
      assert(&FollowingFreeBlock != FreeList && "No tombstone block?");
    }
    FollowingFreeBlock.RemoveFromFreeList();

    // Include the following block into this one.
    BlockSize += FollowingFreeBlock.BlockSize;
    FollowingBlock = &FollowingFreeBlock.getBlockAfter();

    // Tell the block after the block we are coalescing that this block is
    // allocated.
    FollowingBlock->PrevAllocated = 1;
  }

  assert(FollowingBlock->ThisAllocated && "Missed coalescing?");

  if (FreeRangeHeader *PrevFreeBlock = getFreeBlockBefore()) {
    PrevFreeBlock->GrowBlock(PrevFreeBlock->BlockSize + BlockSize);
    return FreeListToReturn ? FreeListToReturn : PrevFreeBlock;
  }

  // Otherwise, mark this block free.
  FreeRangeHeader &FreeBlock = *(FreeRangeHeader*)this;
  FollowingBlock->PrevAllocated = 0;
  FreeBlock.ThisAllocated = 0;

  // Link this into the linked list of free blocks.
  FreeBlock.AddToFreeList(FreeList);

  // Add a marker at the end of the block, indicating the size of this free
  // block.
  FreeBlock.SetEndOfBlockSizeMarker();
  return FreeListToReturn ? FreeListToReturn : &FreeBlock;
}

/// GrowBlock - The block after this block just got deallocated.  Merge it
/// into the current block.
void FreeRangeHeader::GrowBlock(uintptr_t NewSize) {
  assert(NewSize > BlockSize && "Not growing block?");
  BlockSize = NewSize;
  SetEndOfBlockSizeMarker();
  getBlockAfter().PrevAllocated = 0;
}

/// TrimAllocationToSize - If this allocated block is significantly larger
/// than NewSize, split it into two pieces (where the former is NewSize
/// bytes, including the header), and add the new block to the free list.
FreeRangeHeader *MemoryRangeHeader::
TrimAllocationToSize(FreeRangeHeader *FreeList, uint64_t NewSize) {
  assert(ThisAllocated && getBlockAfter().PrevAllocated &&
         "Cannot deallocate part of an allocated block!");

  // Don't allow blocks to be trimmed below minimum required size
  NewSize = std::max<uint64_t>(FreeRangeHeader::getMinBlockSize(), NewSize);

  // Round up size for alignment of header.
  unsigned HeaderAlign = __alignof(FreeRangeHeader);
  NewSize = (NewSize+ (HeaderAlign-1)) & ~(HeaderAlign-1);

  // Size is now the size of the block we will remove from the start of the
  // current block.
  assert(NewSize <= BlockSize &&
         "Allocating more space from this block than exists!");

  // If splitting this block will cause the remainder to be too small, do not
  // split the block.
  if (BlockSize <= NewSize+FreeRangeHeader::getMinBlockSize())
    return FreeList;

  // Otherwise, we splice the required number of bytes out of this block, form
  // a new block immediately after it, then mark this block allocated.
  MemoryRangeHeader &FormerNextBlock = getBlockAfter();

  // Change the size of this block.
  BlockSize = NewSize;

  // Get the new block we just sliced out and turn it into a free block.
  FreeRangeHeader &NewNextBlock = (FreeRangeHeader &)getBlockAfter();
  NewNextBlock.BlockSize = (char*)&FormerNextBlock - (char*)&NewNextBlock;
  NewNextBlock.ThisAllocated = 0;
  NewNextBlock.PrevAllocated = 1;
  NewNextBlock.SetEndOfBlockSizeMarker();
  FormerNextBlock.PrevAllocated = 0;
  NewNextBlock.AddToFreeList(FreeList);
  return &NewNextBlock;
}

//===----------------------------------------------------------------------===//
// Memory Block Implementation.
//===----------------------------------------------------------------------===//

namespace {

  class DefaultJITMemoryManager;

  class JITSlabAllocator : public SlabAllocator {
    DefaultJITMemoryManager &JMM;
  public:
    JITSlabAllocator(DefaultJITMemoryManager &jmm) : JMM(jmm) { }
    virtual ~JITSlabAllocator() { }
    virtual MemSlab *Allocate(size_t Size);
    virtual void Deallocate(MemSlab *Slab);
  };

  /// DefaultJITMemoryManager - Manage memory for the JIT code generation.
  /// This splits a large block of MAP_NORESERVE'd memory into two
  /// sections, one for function stubs, one for the functions themselves.  We
  /// have to do this because we may need to emit a function stub while in the
  /// middle of emitting a function, and we don't know how large the function we
  /// are emitting is.
  class DefaultJITMemoryManager : public JITMemoryManager {

    // Whether to poison freed memory.
    bool PoisonMemory;

    /// LastSlab - This points to the last slab allocated and is used as the
    /// NearBlock parameter to AllocateRWX so that we can attempt to lay out all
    /// stubs, data, and code contiguously in memory.  In general, however, this
    /// is not possible because the NearBlock parameter is ignored on Windows
    /// platforms and even on Unix it works on a best-effort pasis.
    sys::MemoryBlock LastSlab;

    // Memory slabs allocated by the JIT.  We refer to them as slabs so we don't
    // confuse them with the blocks of memory described above.
    std::vector<sys::MemoryBlock> CodeSlabs;
    JITSlabAllocator BumpSlabAllocator;
    BumpPtrAllocator StubAllocator;
    BumpPtrAllocator DataAllocator;

    // Circular list of free blocks.
    FreeRangeHeader *FreeMemoryList;

    // When emitting code into a memory block, this is the block.
    MemoryRangeHeader *CurBlock;

    uint8_t *GOTBase;     // Target Specific reserved memory
  public:
    DefaultJITMemoryManager();
    ~DefaultJITMemoryManager();

    /// allocateNewSlab - Allocates a new MemoryBlock and remembers it as the
    /// last slab it allocated, so that subsequent allocations follow it.
    sys::MemoryBlock allocateNewSlab(size_t size);

    /// DefaultCodeSlabSize - When we have to go map more memory, we allocate at
    /// least this much unless more is requested.
    static const size_t DefaultCodeSlabSize;

    /// DefaultSlabSize - Allocate data into slabs of this size unless we get
    /// an allocation above SizeThreshold.
    static const size_t DefaultSlabSize;

    /// DefaultSizeThreshold - For any allocation larger than this threshold, we
    /// should allocate a separate slab.
    static const size_t DefaultSizeThreshold;

    /// getPointerToNamedFunction - This method returns the address of the
    /// specified function by using the dlsym function call.
    virtual void *getPointerToNamedFunction(const std::string &Name,
                                            bool AbortOnFailure = true);

    void AllocateGOT();

    // Testing methods.
    virtual bool CheckInvariants(std::string &ErrorStr);
    size_t GetDefaultCodeSlabSize() { return DefaultCodeSlabSize; }
    size_t GetDefaultDataSlabSize() { return DefaultSlabSize; }
    size_t GetDefaultStubSlabSize() { return DefaultSlabSize; }
    unsigned GetNumCodeSlabs() { return CodeSlabs.size(); }
    unsigned GetNumDataSlabs() { return DataAllocator.GetNumSlabs(); }
    unsigned GetNumStubSlabs() { return StubAllocator.GetNumSlabs(); }

    /// startFunctionBody - When a function starts, allocate a block of free
    /// executable memory, returning a pointer to it and its actual size.
    uint8_t *startFunctionBody(const Function *F, uintptr_t &ActualSize) {

      FreeRangeHeader* candidateBlock = FreeMemoryList;
      FreeRangeHeader* head = FreeMemoryList;
      FreeRangeHeader* iter = head->Next;

      uintptr_t largest = candidateBlock->BlockSize;

      // Search for the largest free block
      while (iter != head) {
        if (iter->BlockSize > largest) {
          largest = iter->BlockSize;
          candidateBlock = iter;
        }
        iter = iter->Next;
      }

      largest = largest - sizeof(MemoryRangeHeader);

      // If this block isn't big enough for the allocation desired, allocate
      // another block of memory and add it to the free list.
      if (largest < ActualSize ||
          largest <= FreeRangeHeader::getMinBlockSize()) {
        DEBUG(dbgs() << "JIT: Allocating another slab of memory for function.");
        candidateBlock = allocateNewCodeSlab((size_t)ActualSize);
      }

      // Select this candidate block for allocation
      CurBlock = candidateBlock;

      // Allocate the entire memory block.
      FreeMemoryList = candidateBlock->AllocateBlock();
      ActualSize = CurBlock->BlockSize - sizeof(MemoryRangeHeader);
      return (uint8_t *)(CurBlock + 1);
    }

    /// allocateNewCodeSlab - Helper method to allocate a new slab of code
    /// memory from the OS and add it to the free list.  Returns the new
    /// FreeRangeHeader at the base of the slab.
    FreeRangeHeader *allocateNewCodeSlab(size_t MinSize) {
      // If the user needs at least MinSize free memory, then we account for
      // two MemoryRangeHeaders: the one in the user's block, and the one at the
      // end of the slab.
      size_t PaddedMin = MinSize + 2 * sizeof(MemoryRangeHeader);
      size_t SlabSize = std::max(DefaultCodeSlabSize, PaddedMin);
      sys::MemoryBlock B = allocateNewSlab(SlabSize);
      CodeSlabs.push_back(B);
      char *MemBase = (char*)(B.base());

      // Put a tiny allocated block at the end of the memory chunk, so when
      // FreeBlock calls getBlockAfter it doesn't fall off the end.
      MemoryRangeHeader *EndBlock =
          (MemoryRangeHeader*)(MemBase + B.size()) - 1;
      EndBlock->ThisAllocated = 1;
      EndBlock->PrevAllocated = 0;
      EndBlock->BlockSize = sizeof(MemoryRangeHeader);

      // Start out with a vast new block of free memory.
      FreeRangeHeader *NewBlock = (FreeRangeHeader*)MemBase;
      NewBlock->ThisAllocated = 0;
      // Make sure getFreeBlockBefore doesn't look into unmapped memory.
      NewBlock->PrevAllocated = 1;
      NewBlock->BlockSize = (uintptr_t)EndBlock - (uintptr_t)NewBlock;
      NewBlock->SetEndOfBlockSizeMarker();
      NewBlock->AddToFreeList(FreeMemoryList);

      assert(NewBlock->BlockSize - sizeof(MemoryRangeHeader) >= MinSize &&
             "The block was too small!");
      return NewBlock;
    }

    /// endFunctionBody - The function F is now allocated, and takes the memory
    /// in the range [FunctionStart,FunctionEnd).
    void endFunctionBody(const Function *F, uint8_t *FunctionStart,
                         uint8_t *FunctionEnd) {
      assert(FunctionEnd > FunctionStart);
      assert(FunctionStart == (uint8_t *)(CurBlock+1) &&
             "Mismatched function start/end!");

      uintptr_t BlockSize = FunctionEnd - (uint8_t *)CurBlock;

      // Release the memory at the end of this block that isn't needed.
      FreeMemoryList =CurBlock->TrimAllocationToSize(FreeMemoryList, BlockSize);
    }

    /// allocateSpace - Allocate a memory block of the given size.  This method
    /// cannot be called between calls to startFunctionBody and endFunctionBody.
    uint8_t *allocateSpace(intptr_t Size, unsigned Alignment) {
      CurBlock = FreeMemoryList;
      FreeMemoryList = FreeMemoryList->AllocateBlock();

      uint8_t *result = (uint8_t *)(CurBlock + 1);

      if (Alignment == 0) Alignment = 1;
      result = (uint8_t*)(((intptr_t)result+Alignment-1) &
               ~(intptr_t)(Alignment-1));

      uintptr_t BlockSize = result + Size - (uint8_t *)CurBlock;
      FreeMemoryList =CurBlock->TrimAllocationToSize(FreeMemoryList, BlockSize);

      return result;
    }

    /// allocateStub - Allocate memory for a function stub.
    uint8_t *allocateStub(const GlobalValue* F, unsigned StubSize,
                          unsigned Alignment) {
      return (uint8_t*)StubAllocator.Allocate(StubSize, Alignment);
    }

    /// allocateGlobal - Allocate memory for a global.
    uint8_t *allocateGlobal(uintptr_t Size, unsigned Alignment) {
      return (uint8_t*)DataAllocator.Allocate(Size, Alignment);
    }

    /// allocateCodeSection - Allocate memory for a code section.
    uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID) {
      // Grow the required block size to account for the block header
      Size += sizeof(*CurBlock);

      // FIXME: Alignement handling.
      FreeRangeHeader* candidateBlock = FreeMemoryList;
      FreeRangeHeader* head = FreeMemoryList;
      FreeRangeHeader* iter = head->Next;

      uintptr_t largest = candidateBlock->BlockSize;

      // Search for the largest free block.
      while (iter != head) {
        if (iter->BlockSize > largest) {
          largest = iter->BlockSize;
          candidateBlock = iter;
        }
        iter = iter->Next;
      }

      largest = largest - sizeof(MemoryRangeHeader);

      // If this block isn't big enough for the allocation desired, allocate
      // another block of memory and add it to the free list.
      if (largest < Size || largest <= FreeRangeHeader::getMinBlockSize()) {
        DEBUG(dbgs() << "JIT: Allocating another slab of memory for function.");
        candidateBlock = allocateNewCodeSlab((size_t)Size);
      }

      // Select this candidate block for allocation
      CurBlock = candidateBlock;

      // Allocate the entire memory block.
      FreeMemoryList = candidateBlock->AllocateBlock();
      // Release the memory at the end of this block that isn't needed.
      FreeMemoryList = CurBlock->TrimAllocationToSize(FreeMemoryList, Size);
      return (uint8_t *)(CurBlock + 1);
    }

    /// allocateDataSection - Allocate memory for a data section.
    uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID) {
      return (uint8_t*)DataAllocator.Allocate(Size, Alignment);
    }

    /// startExceptionTable - Use startFunctionBody to allocate memory for the
    /// function's exception table.
    uint8_t* startExceptionTable(const Function* F, uintptr_t &ActualSize) {
      return startFunctionBody(F, ActualSize);
    }

    /// endExceptionTable - The exception table of F is now allocated,
    /// and takes the memory in the range [TableStart,TableEnd).
    void endExceptionTable(const Function *F, uint8_t *TableStart,
                           uint8_t *TableEnd, uint8_t* FrameRegister) {
      assert(TableEnd > TableStart);
      assert(TableStart == (uint8_t *)(CurBlock+1) &&
             "Mismatched table start/end!");

      uintptr_t BlockSize = TableEnd - (uint8_t *)CurBlock;

      // Release the memory at the end of this block that isn't needed.
      FreeMemoryList =CurBlock->TrimAllocationToSize(FreeMemoryList, BlockSize);
    }

    uint8_t *getGOTBase() const {
      return GOTBase;
    }

    void deallocateBlock(void *Block) {
      // Find the block that is allocated for this function.
      MemoryRangeHeader *MemRange = static_cast<MemoryRangeHeader*>(Block) - 1;
      assert(MemRange->ThisAllocated && "Block isn't allocated!");

      // Fill the buffer with garbage!
      if (PoisonMemory) {
        memset(MemRange+1, 0xCD, MemRange->BlockSize-sizeof(*MemRange));
      }

      // Free the memory.
      FreeMemoryList = MemRange->FreeBlock(FreeMemoryList);
    }

    /// deallocateFunctionBody - Deallocate all memory for the specified
    /// function body.
    void deallocateFunctionBody(void *Body) {
      if (Body) deallocateBlock(Body);
    }

    /// deallocateExceptionTable - Deallocate memory for the specified
    /// exception table.
    void deallocateExceptionTable(void *ET) {
      if (ET) deallocateBlock(ET);
    }

    /// setMemoryWritable - When code generation is in progress,
    /// the code pages may need permissions changed.
    void setMemoryWritable()
    {
      for (unsigned i = 0, e = CodeSlabs.size(); i != e; ++i)
        sys::Memory::setWritable(CodeSlabs[i]);
    }
    /// setMemoryExecutable - When code generation is done and we're ready to
    /// start execution, the code pages may need permissions changed.
    void setMemoryExecutable()
    {
      for (unsigned i = 0, e = CodeSlabs.size(); i != e; ++i)
        sys::Memory::setExecutable(CodeSlabs[i]);
    }

    /// setPoisonMemory - Controls whether we write garbage over freed memory.
    ///
    void setPoisonMemory(bool poison) {
      PoisonMemory = poison;
    }
  };
}

MemSlab *JITSlabAllocator::Allocate(size_t Size) {
  sys::MemoryBlock B = JMM.allocateNewSlab(Size);
  MemSlab *Slab = (MemSlab*)B.base();
  Slab->Size = B.size();
  Slab->NextPtr = 0;
  return Slab;
}

void JITSlabAllocator::Deallocate(MemSlab *Slab) {
  sys::MemoryBlock B(Slab, Slab->Size);
  sys::Memory::ReleaseRWX(B);
}

DefaultJITMemoryManager::DefaultJITMemoryManager()
  :
#ifdef NDEBUG
    PoisonMemory(false),
#else
    PoisonMemory(true),
#endif
    LastSlab(0, 0),
    BumpSlabAllocator(*this),
    StubAllocator(DefaultSlabSize, DefaultSizeThreshold, BumpSlabAllocator),
    DataAllocator(DefaultSlabSize, DefaultSizeThreshold, BumpSlabAllocator) {

  // Allocate space for code.
  sys::MemoryBlock MemBlock = allocateNewSlab(DefaultCodeSlabSize);
  CodeSlabs.push_back(MemBlock);
  uint8_t *MemBase = (uint8_t*)MemBlock.base();

  // We set up the memory chunk with 4 mem regions, like this:
  //  [ START
  //    [ Free      #0 ] -> Large space to allocate functions from.
  //    [ Allocated #1 ] -> Tiny space to separate regions.
  //    [ Free      #2 ] -> Tiny space so there is always at least 1 free block.
  //    [ Allocated #3 ] -> Tiny space to prevent looking past end of block.
  //  END ]
  //
  // The last three blocks are never deallocated or touched.

  // Add MemoryRangeHeader to the end of the memory region, indicating that
  // the space after the block of memory is allocated.  This is block #3.
  MemoryRangeHeader *Mem3 = (MemoryRangeHeader*)(MemBase+MemBlock.size())-1;
  Mem3->ThisAllocated = 1;
  Mem3->PrevAllocated = 0;
  Mem3->BlockSize     = sizeof(MemoryRangeHeader);

  /// Add a tiny free region so that the free list always has one entry.
  FreeRangeHeader *Mem2 =
    (FreeRangeHeader *)(((char*)Mem3)-FreeRangeHeader::getMinBlockSize());
  Mem2->ThisAllocated = 0;
  Mem2->PrevAllocated = 1;
  Mem2->BlockSize     = FreeRangeHeader::getMinBlockSize();
  Mem2->SetEndOfBlockSizeMarker();
  Mem2->Prev = Mem2;   // Mem2 *is* the free list for now.
  Mem2->Next = Mem2;

  /// Add a tiny allocated region so that Mem2 is never coalesced away.
  MemoryRangeHeader *Mem1 = (MemoryRangeHeader*)Mem2-1;
  Mem1->ThisAllocated = 1;
  Mem1->PrevAllocated = 0;
  Mem1->BlockSize     = sizeof(MemoryRangeHeader);

  // Add a FreeRangeHeader to the start of the function body region, indicating
  // that the space is free.  Mark the previous block allocated so we never look
  // at it.
  FreeRangeHeader *Mem0 = (FreeRangeHeader*)MemBase;
  Mem0->ThisAllocated = 0;
  Mem0->PrevAllocated = 1;
  Mem0->BlockSize = (char*)Mem1-(char*)Mem0;
  Mem0->SetEndOfBlockSizeMarker();
  Mem0->AddToFreeList(Mem2);

  // Start out with the freelist pointing to Mem0.
  FreeMemoryList = Mem0;

  GOTBase = NULL;
}

void DefaultJITMemoryManager::AllocateGOT() {
  assert(GOTBase == 0 && "Cannot allocate the got multiple times");
  GOTBase = new uint8_t[sizeof(void*) * 8192];
  HasGOT = true;
}

DefaultJITMemoryManager::~DefaultJITMemoryManager() {
  for (unsigned i = 0, e = CodeSlabs.size(); i != e; ++i)
    sys::Memory::ReleaseRWX(CodeSlabs[i]);

  delete[] GOTBase;
}

sys::MemoryBlock DefaultJITMemoryManager::allocateNewSlab(size_t size) {
  // Allocate a new block close to the last one.
  std::string ErrMsg;
  sys::MemoryBlock *LastSlabPtr = LastSlab.base() ? &LastSlab : 0;
  sys::MemoryBlock B = sys::Memory::AllocateRWX(size, LastSlabPtr, &ErrMsg);
  if (B.base() == 0) {
    report_fatal_error("Allocation failed when allocating new memory in the"
                       " JIT\n" + Twine(ErrMsg));
  }
  LastSlab = B;
  ++NumSlabs;
  // Initialize the slab to garbage when debugging.
  if (PoisonMemory) {
    memset(B.base(), 0xCD, B.size());
  }
  return B;
}

/// CheckInvariants - For testing only.  Return "" if all internal invariants
/// are preserved, and a helpful error message otherwise.  For free and
/// allocated blocks, make sure that adding BlockSize gives a valid block.
/// For free blocks, make sure they're in the free list and that their end of
/// block size marker is correct.  This function should return an error before
/// accessing bad memory.  This function is defined here instead of in
/// JITMemoryManagerTest.cpp so that we don't have to expose all of the
/// implementation details of DefaultJITMemoryManager.
bool DefaultJITMemoryManager::CheckInvariants(std::string &ErrorStr) {
  raw_string_ostream Err(ErrorStr);

  // Construct a the set of FreeRangeHeader pointers so we can query it
  // efficiently.
  llvm::SmallPtrSet<MemoryRangeHeader*, 16> FreeHdrSet;
  FreeRangeHeader* FreeHead = FreeMemoryList;
  FreeRangeHeader* FreeRange = FreeHead;

  do {
    // Check that the free range pointer is in the blocks we've allocated.
    bool Found = false;
    for (std::vector<sys::MemoryBlock>::iterator I = CodeSlabs.begin(),
         E = CodeSlabs.end(); I != E && !Found; ++I) {
      char *Start = (char*)I->base();
      char *End = Start + I->size();
      Found = (Start <= (char*)FreeRange && (char*)FreeRange < End);
    }
    if (!Found) {
      Err << "Corrupt free list; points to " << FreeRange;
      return false;
    }

    if (FreeRange->Next->Prev != FreeRange) {
      Err << "Next and Prev pointers do not match.";
      return false;
    }

    // Otherwise, add it to the set.
    FreeHdrSet.insert(FreeRange);
    FreeRange = FreeRange->Next;
  } while (FreeRange != FreeHead);

  // Go over each block, and look at each MemoryRangeHeader.
  for (std::vector<sys::MemoryBlock>::iterator I = CodeSlabs.begin(),
       E = CodeSlabs.end(); I != E; ++I) {
    char *Start = (char*)I->base();
    char *End = Start + I->size();

    // Check each memory range.
    for (MemoryRangeHeader *Hdr = (MemoryRangeHeader*)Start, *LastHdr = NULL;
         Start <= (char*)Hdr && (char*)Hdr < End;
         Hdr = &Hdr->getBlockAfter()) {
      if (Hdr->ThisAllocated == 0) {
        // Check that this range is in the free list.
        if (!FreeHdrSet.count(Hdr)) {
          Err << "Found free header at " << Hdr << " that is not in free list.";
          return false;
        }

        // Now make sure the size marker at the end of the block is correct.
        uintptr_t *Marker = ((uintptr_t*)&Hdr->getBlockAfter()) - 1;
        if (!(Start <= (char*)Marker && (char*)Marker < End)) {
          Err << "Block size in header points out of current MemoryBlock.";
          return false;
        }
        if (Hdr->BlockSize != *Marker) {
          Err << "End of block size marker (" << *Marker << ") "
              << "and BlockSize (" << Hdr->BlockSize << ") don't match.";
          return false;
        }
      }

      if (LastHdr && LastHdr->ThisAllocated != Hdr->PrevAllocated) {
        Err << "Hdr->PrevAllocated (" << Hdr->PrevAllocated << ") != "
            << "LastHdr->ThisAllocated (" << LastHdr->ThisAllocated << ")";
        return false;
      } else if (!LastHdr && !Hdr->PrevAllocated) {
        Err << "The first header should have PrevAllocated true.";
        return false;
      }

      // Remember the last header.
      LastHdr = Hdr;
    }
  }

  // All invariants are preserved.
  return true;
}

//===----------------------------------------------------------------------===//
// getPointerToNamedFunction() implementation.
//===----------------------------------------------------------------------===//

// AtExitHandlers - List of functions to call when the program exits,
// registered with the atexit() library function.
static std::vector<void (*)()> AtExitHandlers;

/// runAtExitHandlers - Run any functions registered by the program's
/// calls to atexit(3), which we intercept and store in
/// AtExitHandlers.
///
static void runAtExitHandlers() {
  while (!AtExitHandlers.empty()) {
    void (*Fn)() = AtExitHandlers.back();
    AtExitHandlers.pop_back();
    Fn();
  }
}

//===----------------------------------------------------------------------===//
// Function stubs that are invoked instead of certain library calls
//
// Force the following functions to be linked in to anything that uses the
// JIT. This is a hack designed to work around the all-too-clever Glibc
// strategy of making these functions work differently when inlined vs. when
// not inlined, and hiding their real definitions in a separate archive file
// that the dynamic linker can't see. For more info, search for
// 'libc_nonshared.a' on Google, or read http://llvm.org/PR274.
#if defined(__linux__)
/* stat functions are redirecting to __xstat with a version number.  On x86-64
 * linking with libc_nonshared.a and -Wl,--export-dynamic doesn't make 'stat'
 * available as an exported symbol, so we have to add it explicitly.
 */
namespace {
class StatSymbols {
public:
  StatSymbols() {
    sys::DynamicLibrary::AddSymbol("stat", (void*)(intptr_t)stat);
    sys::DynamicLibrary::AddSymbol("fstat", (void*)(intptr_t)fstat);
    sys::DynamicLibrary::AddSymbol("lstat", (void*)(intptr_t)lstat);
    sys::DynamicLibrary::AddSymbol("stat64", (void*)(intptr_t)stat64);
    sys::DynamicLibrary::AddSymbol("\x1stat64", (void*)(intptr_t)stat64);
    sys::DynamicLibrary::AddSymbol("\x1open64", (void*)(intptr_t)open64);
    sys::DynamicLibrary::AddSymbol("\x1lseek64", (void*)(intptr_t)lseek64);
    sys::DynamicLibrary::AddSymbol("fstat64", (void*)(intptr_t)fstat64);
    sys::DynamicLibrary::AddSymbol("lstat64", (void*)(intptr_t)lstat64);
    sys::DynamicLibrary::AddSymbol("atexit", (void*)(intptr_t)atexit);
    sys::DynamicLibrary::AddSymbol("mknod", (void*)(intptr_t)mknod);
  }
};
}
static StatSymbols initStatSymbols;
#endif // __linux__

// jit_exit - Used to intercept the "exit" library call.
static void jit_exit(int Status) {
  runAtExitHandlers();   // Run atexit handlers...
  exit(Status);
}

// jit_atexit - Used to intercept the "atexit" library call.
static int jit_atexit(void (*Fn)()) {
  AtExitHandlers.push_back(Fn);    // Take note of atexit handler...
  return 0;  // Always successful
}

static int jit_noop() {
  return 0;
}

//===----------------------------------------------------------------------===//
//
/// getPointerToNamedFunction - This method returns the address of the specified
/// function by using the dynamic loader interface.  As such it is only useful
/// for resolving library symbols, not code generated symbols.
///
void *DefaultJITMemoryManager::getPointerToNamedFunction(const std::string &Name,
                                                         bool AbortOnFailure) {
  // Check to see if this is one of the functions we want to intercept.  Note,
  // we cast to intptr_t here to silence a -pedantic warning that complains
  // about casting a function pointer to a normal pointer.
  if (Name == "exit") return (void*)(intptr_t)&jit_exit;
  if (Name == "atexit") return (void*)(intptr_t)&jit_atexit;

  // We should not invoke parent's ctors/dtors from generated main()!
  // On Mingw and Cygwin, the symbol __main is resolved to
  // callee's(eg. tools/lli) one, to invoke wrong duplicated ctors
  // (and register wrong callee's dtors with atexit(3)).
  // We expect ExecutionEngine::runStaticConstructorsDestructors()
  // is called before ExecutionEngine::runFunctionAsMain() is called.
  if (Name == "__main") return (void*)(intptr_t)&jit_noop;

  const char *NameStr = Name.c_str();
  // If this is an asm specifier, skip the sentinal.
  if (NameStr[0] == 1) ++NameStr;

  // If it's an external function, look it up in the process image...
  void *Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr);
  if (Ptr) return Ptr;

  // If it wasn't found and if it starts with an underscore ('_') character,
  // try again without the underscore.
  if (NameStr[0] == '_') {
    Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr+1);
    if (Ptr) return Ptr;
  }

  // Darwin/PPC adds $LDBLStub suffixes to various symbols like printf.  These
  // are references to hidden visibility symbols that dlsym cannot resolve.
  // If we have one of these, strip off $LDBLStub and try again.
#if defined(__APPLE__) && defined(__ppc__)
  if (Name.size() > 9 && Name[Name.size()-9] == '$' &&
      memcmp(&Name[Name.size()-8], "LDBLStub", 8) == 0) {
    // First try turning $LDBLStub into $LDBL128. If that fails, strip it off.
    // This mirrors logic in libSystemStubs.a.
    std::string Prefix = std::string(Name.begin(), Name.end()-9);
    if (void *Ptr = getPointerToNamedFunction(Prefix+"$LDBL128", false))
      return Ptr;
    if (void *Ptr = getPointerToNamedFunction(Prefix, false))
      return Ptr;
  }
#endif

  if (AbortOnFailure) {
    report_fatal_error("Program used external function '"+Name+
                      "' which could not be resolved!");
  }
  return 0;
}



JITMemoryManager *JITMemoryManager::CreateDefaultMemManager() {
  return new DefaultJITMemoryManager();
}

// Allocate memory for code in 512K slabs.
const size_t DefaultJITMemoryManager::DefaultCodeSlabSize = 512 * 1024;

// Allocate globals and stubs in slabs of 64K.  (probably 16 pages)
const size_t DefaultJITMemoryManager::DefaultSlabSize = 64 * 1024;

// Waste at most 16K at the end of each bump slab.  (probably 4 pages)
const size_t DefaultJITMemoryManager::DefaultSizeThreshold = 16 * 1024;
