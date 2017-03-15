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
  int *a = (int*)Alloc.Allocate(sizeof(int), alignof(int));
  int *b = (int*)Alloc.Allocate(sizeof(int) * 10, alignof(int));
  int *c = (int*)Alloc.Allocate(sizeof(int), alignof(int));
  *a = 1;
  b[0] = 2;
  b[9] = 2;
  *c = 3;
  EXPECT_EQ(1, *a);
  EXPECT_EQ(2, b[0]);
  EXPECT_EQ(2, b[9]);
  EXPECT_EQ(3, *c);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());

  BumpPtrAllocator Alloc2 = std::move(Alloc);
  EXPECT_EQ(0U, Alloc.GetNumSlabs());
  EXPECT_EQ(1U, Alloc2.GetNumSlabs());

  // Make sure the old pointers still work. These are especially interesting
  // under ASan or Valgrind.
  EXPECT_EQ(1, *a);
  EXPECT_EQ(2, b[0]);
  EXPECT_EQ(2, b[9]);
  EXPECT_EQ(3, *c);

  Alloc = std::move(Alloc2);
  EXPECT_EQ(0U, Alloc2.GetNumSlabs());
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
}

// Allocate enough bytes to create three slabs.
TEST(AllocatorTest, ThreeSlabs) {
  BumpPtrAllocator Alloc;
  Alloc.Allocate(3000, 1);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
  Alloc.Allocate(3000, 1);
  EXPECT_EQ(2U, Alloc.GetNumSlabs());
  Alloc.Allocate(3000, 1);
  EXPECT_EQ(3U, Alloc.GetNumSlabs());
}

// Allocate enough bytes to create two slabs, reset the allocator, and do it
// again.
TEST(AllocatorTest, TestReset) {
  BumpPtrAllocator Alloc;

  // Allocate something larger than the SizeThreshold=4096.
  (void)Alloc.Allocate(5000, 1);
  Alloc.Reset();
  // Calling Reset should free all CustomSizedSlabs.
  EXPECT_EQ(0u, Alloc.GetNumSlabs());

  Alloc.Allocate(3000, 1);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
  Alloc.Allocate(3000, 1);
  EXPECT_EQ(2U, Alloc.GetNumSlabs());
  Alloc.Reset();
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
  Alloc.Allocate(3000, 1);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
  Alloc.Allocate(3000, 1);
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
  BumpPtrAllocator Alloc;

  // Fill the slab right up until the end pointer.
  Alloc.Allocate(4096, 1);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());

  // If we don't allocate a new slab, then we will have overflowed.
  Alloc.Allocate(1, 1);
  EXPECT_EQ(2U, Alloc.GetNumSlabs());
}

// Test allocating with a size larger than the initial slab size.
TEST(AllocatorTest, TestSmallSlabSize) {
  BumpPtrAllocator Alloc;

  Alloc.Allocate(8000, 1);
  EXPECT_EQ(1U, Alloc.GetNumSlabs());
}

// Test requesting alignment that goes past the end of the current slab.
TEST(AllocatorTest, TestAlignmentPastSlab) {
  BumpPtrAllocator Alloc;
  Alloc.Allocate(4095, 1);

  // Aligning the current slab pointer is likely to move it past the end of the
  // slab, which would confuse any unsigned comparisons with the difference of
  // the end pointer and the aligned pointer.
  Alloc.Allocate(1024, 8192);

  EXPECT_EQ(2U, Alloc.GetNumSlabs());
}

// Mock slab allocator that returns slabs aligned on 4096 bytes.  There is no
// easy portable way to do this, so this is kind of a hack.
class MockSlabAllocator {
  static size_t LastSlabSize;

public:
  ~MockSlabAllocator() { }

  void *Allocate(size_t Size, size_t /*Alignment*/) {
    // Allocate space for the alignment, the slab, and a void* that goes right
    // before the slab.
    size_t Alignment = 4096;
    void *MemBase = malloc(Size + Alignment - 1 + sizeof(void*));

    // Find the slab start.
    void *Slab = (void *)alignAddr((char*)MemBase + sizeof(void *), Alignment);

    // Hold a pointer to the base so we can free the whole malloced block.
    ((void**)Slab)[-1] = MemBase;

    LastSlabSize = Size;
    return Slab;
  }

  void Deallocate(void *Slab, size_t Size) {
    free(((void**)Slab)[-1]);
  }

  static size_t GetLastSlabSize() { return LastSlabSize; }
};

size_t MockSlabAllocator::LastSlabSize = 0;

// Allocate a large-ish block with a really large alignment so that the
// allocator will think that it has space, but after it does the alignment it
// will not.
TEST(AllocatorTest, TestBigAlignment) {
  BumpPtrAllocatorImpl<MockSlabAllocator> Alloc;

  // First allocate a tiny bit to ensure we have to re-align things.
  (void)Alloc.Allocate(1, 1);

  // Now the big chunk with a big alignment.
  (void)Alloc.Allocate(3000, 2048);

  // We test that the last slab size is not the default 4096 byte slab, but
  // rather a custom sized slab that is larger.
  EXPECT_GT(MockSlabAllocator::GetLastSlabSize(), 4096u);
}

}  // anonymous namespace
