//===- llvm/unittest/Support/AllocatorTest.cpp - BumpPtrAllocator tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// Test zero-sized allocations.
// In general we don't need to allocate memory for these.
// However Allocate never returns null, so if the first allocation is zero-sized
// we end up creating a slab for it.
TEST(AllocatorTest, TestZero) {
  BumpPtrAllocator Alloc;
  Alloc.setRedZoneSize(0); // else our arithmetic is all off
  EXPECT_EQ(0u, Alloc.GetNumSlabs());
  EXPECT_EQ(0u, Alloc.getBytesAllocated());

  void *Empty = Alloc.Allocate(0, 1);
  EXPECT_NE(Empty, nullptr) << "Allocate is __attribute__((returns_nonnull))";
  EXPECT_EQ(1u, Alloc.GetNumSlabs()) << "Allocated a slab to point to";
  EXPECT_EQ(0u, Alloc.getBytesAllocated());

  void *Large = Alloc.Allocate(4096, 1);
  EXPECT_EQ(1u, Alloc.GetNumSlabs());
  EXPECT_EQ(4096u, Alloc.getBytesAllocated());
  EXPECT_EQ(Empty, Large);

  void *Empty2 = Alloc.Allocate(0, 1);
  EXPECT_NE(Empty2, nullptr);
  EXPECT_EQ(1u, Alloc.GetNumSlabs());
  EXPECT_EQ(4096u, Alloc.getBytesAllocated());
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

// Test allocating with a decreased growth delay.
TEST(AllocatorTest, TestFasterSlabGrowthDelay) {
  const size_t SlabSize = 4096;
  // Decrease the growth delay to double the slab size every slab.
  const size_t GrowthDelay = 1;
  BumpPtrAllocatorImpl<MallocAllocator, SlabSize, SlabSize, GrowthDelay> Alloc;
  // Disable the red zone for this test. The additional bytes allocated for the
  // red zone would change the allocation numbers we check below.
  Alloc.setRedZoneSize(0);

  Alloc.Allocate(SlabSize, 1);
  EXPECT_EQ(SlabSize, Alloc.getTotalMemory());
  // We hit our growth delay with the previous allocation so the next
  // allocation should get a twice as large slab.
  Alloc.Allocate(SlabSize, 1);
  EXPECT_EQ(SlabSize * 3, Alloc.getTotalMemory());
  Alloc.Allocate(SlabSize, 1);
  EXPECT_EQ(SlabSize * 3, Alloc.getTotalMemory());

  // Both slabs are full again and hit the growth delay again, so the
  // next allocation should again get a slab with four times the size of the
  // original slab size. In total we now should have a memory size of:
  // 1 + 2 + 4 * SlabSize.
  Alloc.Allocate(SlabSize, 1);
  EXPECT_EQ(SlabSize * 7, Alloc.getTotalMemory());
}

// Test allocating with a increased growth delay.
TEST(AllocatorTest, TestSlowerSlabGrowthDelay) {
  const size_t SlabSize = 16;
  // Increase the growth delay to only double the slab size every 256 slabs.
  const size_t GrowthDelay = 256;
  BumpPtrAllocatorImpl<MallocAllocator, SlabSize, SlabSize, GrowthDelay> Alloc;
  // Disable the red zone for this test. The additional bytes allocated for the
  // red zone would change the allocation numbers we check below.
  Alloc.setRedZoneSize(0);

  // Allocate 256 slabs. We should keep getting slabs with the original size
  // as we haven't hit our growth delay on the last allocation.
  for (std::size_t i = 0; i < GrowthDelay; ++i)
    Alloc.Allocate(SlabSize, 1);
  EXPECT_EQ(SlabSize * GrowthDelay, Alloc.getTotalMemory());
  // Allocate another slab. This time we should get another slab allocated
  // that is twice as large as the normal slab size.
  Alloc.Allocate(SlabSize, 1);
  EXPECT_EQ(SlabSize * GrowthDelay + SlabSize * 2, Alloc.getTotalMemory());
}

TEST(AllocatorTest, TestIdentifyObject) {
  BumpPtrAllocator Alloc;

  uint64_t *a = (uint64_t *)Alloc.Allocate(sizeof(uint64_t), alignof(uint64_t));
  std::optional<int64_t> maybe_a_belongs = Alloc.identifyObject(a);
  EXPECT_TRUE(maybe_a_belongs.has_value());
  EXPECT_TRUE(*maybe_a_belongs >= 0);

  uint64_t *b = nullptr;
  std::optional<int64_t> maybe_b_belongs = Alloc.identifyObject(b);
  EXPECT_FALSE(maybe_b_belongs);

  // The default slab size is 4096 (or 512 uint64_t values). Custom slabs are
  // allocated when the requested size is larger than the slab size.
  uint64_t *c =
      (uint64_t *)Alloc.Allocate(sizeof(uint64_t) * 1024, alignof(uint64_t));
  std::optional<int64_t> maybe_c_belongs = Alloc.identifyObject(c);
  EXPECT_TRUE(maybe_c_belongs.has_value());
  EXPECT_TRUE(*maybe_c_belongs < 0);
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
    Align Alignment(4096);
    void *MemBase = safe_malloc(Size + Alignment.value() - 1 + sizeof(void *));

    // Find the slab start.
    void *Slab = (void *)alignAddr((char*)MemBase + sizeof(void *), Alignment);

    // Hold a pointer to the base so we can free the whole malloced block.
    ((void**)Slab)[-1] = MemBase;

    LastSlabSize = Size;
    return Slab;
  }

  void Deallocate(void *Slab, size_t /*Size*/, size_t /*Alignment*/) {
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
