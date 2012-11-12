//===- llvm/unittest/Support/AllocatorTest.cpp - BumpPtrAllocator tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Allocator.h"

#include "gtest/gtest.h"
#include <cstdlib>

using namespace llvm;

namespace {

TEST(AllocatorTest, Basics) {
  BumpPtrAllocator Alloc;
  int *a = (int*)Alloc.Allocate(sizeof(int), 0);
  int *b = (int*)Alloc.Allocate(sizeof(int) * 10, 0);
  int *c = (int*)Alloc.Allocate(sizeof(int), 0);
  *a = 1;
  b[0] = 2;
  b[9] = 2;
  *c = 3;
  EXPECT_EQ(1, *a);
  EXPECT_EQ(2, b[0]);
  EXPECT_EQ(2, b[9]);
  EXPECT_EQ(3, *c);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
}

// Allocate enough bytes to create three slabs.
TEST(AllocatorTest, ThreeSlabs) {
  BumpPtrAllocator Alloc(4096, 4096);
  Alloc.Allocate(3000, 0);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
  Alloc.Allocate(3000, 0);
  EXPECT_EQ(2U, Alloc.GetNumSlabs());
  Alloc.Allocate(3000, 0);
  EXPECT_EQ(3U, Alloc.GetNumSlabs());
}

// Allocate enough bytes to create two slabs, reset the allocator, and do it
// again.
TEST(AllocatorTest, TestReset) {
  BumpPtrAllocator Alloc(4096, 4096);
  Alloc.Allocate(3000, 0);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
  Alloc.Allocate(3000, 0);
  EXPECT_EQ(2U, Alloc.GetNumSlabs());
  Alloc.Reset();
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
  Alloc.Allocate(3000, 0);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
  Alloc.Allocate(3000, 0);
  EXPECT_EQ(2U, Alloc.GetNumSlabs());
}

// Test some allocations at varying alignments.
TEST(AllocatorTest, TestAlignment) {
  BumpPtrAllocator Alloc;
  uintptr_t a;
  a = (uintptr_t)Alloc.Allocate(1, 2);
  EXPECT_EQ(0U, a & 1);
  a = (uintptr_t)Alloc.Allocate(1, 4);
  EXPECT_EQ(0U, a & 3);
  a = (uintptr_t)Alloc.Allocate(1, 8);
  EXPECT_EQ(0U, a & 7);
  a = (uintptr_t)Alloc.Allocate(1, 16);
  EXPECT_EQ(0U, a & 15);
  a = (uintptr_t)Alloc.Allocate(1, 32);
  EXPECT_EQ(0U, a & 31);
  a = (uintptr_t)Alloc.Allocate(1, 64);
  EXPECT_EQ(0U, a & 63);
  a = (uintptr_t)Alloc.Allocate(1, 128);
  EXPECT_EQ(0U, a & 127);
}

// Test allocating just over the slab size.  This tests a bug where before the
// allocator incorrectly calculated the buffer end pointer.
TEST(AllocatorTest, TestOverflow) {
  BumpPtrAllocator Alloc(4096, 4096);

  // Fill the slab right up until the end pointer.
  Alloc.Allocate(4096 - sizeof(MemSlab), 0);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());

  // If we don't allocate a new slab, then we will have overflowed.
  Alloc.Allocate(1, 0);
  EXPECT_EQ(2U, Alloc.GetNumSlabs());
}

// Test allocating with a size larger than the initial slab size.
TEST(AllocatorTest, TestSmallSlabSize) {
  BumpPtrAllocator Alloc(128);

  Alloc.Allocate(200, 0);
  EXPECT_EQ(2U, Alloc.GetNumSlabs());
}

// Mock slab allocator that returns slabs aligned on 4096 bytes.  There is no
// easy portable way to do this, so this is kind of a hack.
class MockSlabAllocator : public SlabAllocator {
  MemSlab *LastSlab;

public:
  virtual ~MockSlabAllocator() { }

  virtual MemSlab *Allocate(size_t Size) {
    // Allocate space for the alignment, the slab, and a void* that goes right
    // before the slab.
    size_t Alignment = 4096;
    void *MemBase = malloc(Size + Alignment - 1 + sizeof(void*));

    // Make the slab.
    MemSlab *Slab = (MemSlab*)(((uintptr_t)MemBase+sizeof(void*)+Alignment-1) &
                               ~(uintptr_t)(Alignment - 1));
    Slab->Size = Size;
    Slab->NextPtr = 0;

    // Hold a pointer to the base so we can free the whole malloced block.
    ((void**)Slab)[-1] = MemBase;

    LastSlab = Slab;
    return Slab;
  }

  virtual void Deallocate(MemSlab *Slab) {
    free(((void**)Slab)[-1]);
  }

  MemSlab *GetLastSlab() {
    return LastSlab;
  }
};

// Allocate a large-ish block with a really large alignment so that the
// allocator will think that it has space, but after it does the alignment it
// will not.
TEST(AllocatorTest, TestBigAlignment) {
  MockSlabAllocator SlabAlloc;
  BumpPtrAllocator Alloc(4096, 4096, SlabAlloc);
  uintptr_t Ptr = (uintptr_t)Alloc.Allocate(3000, 2048);
  MemSlab *Slab = SlabAlloc.GetLastSlab();
  EXPECT_LE(Ptr + 3000, ((uintptr_t)Slab) + Slab->Size);
}

}  // anonymous namespace
