//===- MSFBuilderTest.cpp  Tests manipulation of MSF stream metadata ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::msf;
using namespace testing;

namespace {
class MSFBuilderTest : public testing::Test {
protected:
  void initializeSimpleSuperBlock(msf::SuperBlock &SB) {
    initializeSuperBlock(SB);
    SB.NumBlocks = 1000;
    SB.NumDirectoryBytes = 8192;
  }

  void initializeSuperBlock(msf::SuperBlock &SB) {
    ::memset(&SB, 0, sizeof(SB));

    ::memcpy(SB.MagicBytes, msf::Magic, sizeof(msf::Magic));
    SB.FreeBlockMapBlock = 1;
    SB.BlockMapAddr = 1;
    SB.BlockSize = 4096;
    SB.NumDirectoryBytes = 0;
    SB.NumBlocks = 2; // one for the Super Block, one for the directory
  }

  BumpPtrAllocator Allocator;
};
} // namespace

TEST_F(MSFBuilderTest, ValidateSuperBlockAccept) {
  // Test that a known good super block passes validation.
  SuperBlock SB;
  initializeSuperBlock(SB);

  EXPECT_THAT_ERROR(msf::validateSuperBlock(SB), Succeeded());
}

TEST_F(MSFBuilderTest, ValidateSuperBlockReject) {
  // Test that various known problems cause a super block to be rejected.
  SuperBlock SB;
  initializeSimpleSuperBlock(SB);

  // Mismatched magic
  SB.MagicBytes[0] = 8;
  EXPECT_THAT_ERROR(msf::validateSuperBlock(SB), Failed());
  initializeSimpleSuperBlock(SB);

  // Block 0 is reserved for super block, can't be occupied by the block map
  SB.BlockMapAddr = 0;
  EXPECT_THAT_ERROR(msf::validateSuperBlock(SB), Failed());
  initializeSimpleSuperBlock(SB);

  // Block sizes have to be powers of 2.
  SB.BlockSize = 3120;
  EXPECT_THAT_ERROR(msf::validateSuperBlock(SB), Failed());
  initializeSimpleSuperBlock(SB);

  // The directory itself has a maximum size.
  SB.NumDirectoryBytes = SB.BlockSize * SB.BlockSize / 4;
  EXPECT_THAT_ERROR(msf::validateSuperBlock(SB), Succeeded());
  SB.NumDirectoryBytes = SB.NumDirectoryBytes + 4;
  EXPECT_THAT_ERROR(msf::validateSuperBlock(SB), Failed());
}

TEST_F(MSFBuilderTest, TestUsedBlocksMarkedAsUsed) {
  // Test that when assigning a stream to a known list of blocks, the blocks
  // are correctly marked as used after adding, but no other incorrect blocks
  // are accidentally marked as used.

  std::vector<uint32_t> Blocks = {4, 5, 6, 7, 8, 9, 10, 11, 12};
  // Allocate some extra blocks at the end so we can verify that they're free
  // after the initialization.
  uint32_t NumBlocks = msf::getMinimumBlockCount() + Blocks.size() + 10;
  auto ExpectedMsf = MSFBuilder::create(Allocator, 4096, NumBlocks);
  ASSERT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  EXPECT_THAT_EXPECTED(Msf.addStream(Blocks.size() * 4096, Blocks),
                       Succeeded());

  for (auto B : Blocks) {
    EXPECT_FALSE(Msf.isBlockFree(B));
  }

  uint32_t FreeBlockStart = Blocks.back() + 1;
  for (uint32_t I = FreeBlockStart; I < NumBlocks; ++I) {
    EXPECT_TRUE(Msf.isBlockFree(I));
  }
}

TEST_F(MSFBuilderTest, TestAddStreamNoDirectoryBlockIncrease) {
  // Test that adding a new stream correctly updates the directory.  This only
  // tests the case where the directory *DOES NOT* grow large enough that it
  // crosses a Block boundary.
  auto ExpectedMsf = MSFBuilder::create(Allocator, 4096);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  auto ExpectedL1 = Msf.generateLayout();
  EXPECT_THAT_EXPECTED(ExpectedL1, Succeeded());
  MSFLayout &L1 = *ExpectedL1;

  auto OldDirBlocks = L1.DirectoryBlocks;
  EXPECT_EQ(1U, OldDirBlocks.size());

  auto ExpectedMsf2 = MSFBuilder::create(Allocator, 4096);
  EXPECT_THAT_EXPECTED(ExpectedMsf2, Succeeded());
  auto &Msf2 = *ExpectedMsf2;

  EXPECT_THAT_EXPECTED(Msf2.addStream(4000), Succeeded());
  EXPECT_EQ(1U, Msf2.getNumStreams());
  EXPECT_EQ(4000U, Msf2.getStreamSize(0));
  auto Blocks = Msf2.getStreamBlocks(0);
  EXPECT_EQ(1U, Blocks.size());

  auto ExpectedL2 = Msf2.generateLayout();
  EXPECT_THAT_EXPECTED(ExpectedL2, Succeeded());
  MSFLayout &L2 = *ExpectedL2;
  auto NewDirBlocks = L2.DirectoryBlocks;
  EXPECT_EQ(1U, NewDirBlocks.size());
}

TEST_F(MSFBuilderTest, TestAddStreamWithDirectoryBlockIncrease) {
  // Test that adding a new stream correctly updates the directory.  This only
  // tests the case where the directory *DOES* grow large enough that it
  // crosses a Block boundary.  This is because the newly added stream occupies
  // so many Blocks that need to be indexed in the directory that the directory
  // crosses a Block boundary.
  auto ExpectedMsf = MSFBuilder::create(Allocator, 4096);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  EXPECT_THAT_EXPECTED(Msf.addStream(4096 * 4096 / sizeof(uint32_t)),
                       Succeeded());

  auto ExpectedL1 = Msf.generateLayout();
  EXPECT_THAT_EXPECTED(ExpectedL1, Succeeded());
  MSFLayout &L1 = *ExpectedL1;
  auto DirBlocks = L1.DirectoryBlocks;
  EXPECT_EQ(2U, DirBlocks.size());
}

TEST_F(MSFBuilderTest, TestGrowStreamNoBlockIncrease) {
  // Test growing an existing stream by a value that does not affect the number
  // of blocks it occupies.
  auto ExpectedMsf = MSFBuilder::create(Allocator, 4096);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  EXPECT_THAT_EXPECTED(Msf.addStream(1024), Succeeded());
  EXPECT_EQ(1024U, Msf.getStreamSize(0));
  auto OldStreamBlocks = Msf.getStreamBlocks(0);
  EXPECT_EQ(1U, OldStreamBlocks.size());

  EXPECT_THAT_ERROR(Msf.setStreamSize(0, 2048), Succeeded());
  EXPECT_EQ(2048U, Msf.getStreamSize(0));
  auto NewStreamBlocks = Msf.getStreamBlocks(0);
  EXPECT_EQ(1U, NewStreamBlocks.size());

  EXPECT_EQ(OldStreamBlocks, NewStreamBlocks);
}

TEST_F(MSFBuilderTest, TestGrowStreamWithBlockIncrease) {
  // Test that growing an existing stream to a value large enough that it causes
  // the need to allocate new Blocks to the stream correctly updates the
  // stream's
  // block list.
  auto ExpectedMsf = MSFBuilder::create(Allocator, 4096);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  EXPECT_THAT_EXPECTED(Msf.addStream(2048), Succeeded());
  EXPECT_EQ(2048U, Msf.getStreamSize(0));
  std::vector<uint32_t> OldStreamBlocks = Msf.getStreamBlocks(0);
  EXPECT_EQ(1U, OldStreamBlocks.size());

  EXPECT_THAT_ERROR(Msf.setStreamSize(0, 6144), Succeeded());
  EXPECT_EQ(6144U, Msf.getStreamSize(0));
  std::vector<uint32_t> NewStreamBlocks = Msf.getStreamBlocks(0);
  EXPECT_EQ(2U, NewStreamBlocks.size());

  EXPECT_EQ(OldStreamBlocks[0], NewStreamBlocks[0]);
  EXPECT_NE(NewStreamBlocks[0], NewStreamBlocks[1]);
}

TEST_F(MSFBuilderTest, TestShrinkStreamNoBlockDecrease) {
  // Test that shrinking an existing stream by a value that does not affect the
  // number of Blocks it occupies makes no changes to stream's block list.
  auto ExpectedMsf = MSFBuilder::create(Allocator, 4096);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  EXPECT_THAT_EXPECTED(Msf.addStream(2048), Succeeded());
  EXPECT_EQ(2048U, Msf.getStreamSize(0));
  std::vector<uint32_t> OldStreamBlocks = Msf.getStreamBlocks(0);
  EXPECT_EQ(1U, OldStreamBlocks.size());

  EXPECT_THAT_ERROR(Msf.setStreamSize(0, 1024), Succeeded());
  EXPECT_EQ(1024U, Msf.getStreamSize(0));
  std::vector<uint32_t> NewStreamBlocks = Msf.getStreamBlocks(0);
  EXPECT_EQ(1U, NewStreamBlocks.size());

  EXPECT_EQ(OldStreamBlocks, NewStreamBlocks);
}

TEST_F(MSFBuilderTest, TestShrinkStreamWithBlockDecrease) {
  // Test that shrinking an existing stream to a value large enough that it
  // causes the need to deallocate new Blocks to the stream correctly updates
  // the stream's block list.
  auto ExpectedMsf = MSFBuilder::create(Allocator, 4096);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  EXPECT_THAT_EXPECTED(Msf.addStream(6144), Succeeded());
  EXPECT_EQ(6144U, Msf.getStreamSize(0));
  std::vector<uint32_t> OldStreamBlocks = Msf.getStreamBlocks(0);
  EXPECT_EQ(2U, OldStreamBlocks.size());

  EXPECT_THAT_ERROR(Msf.setStreamSize(0, 2048), Succeeded());
  EXPECT_EQ(2048U, Msf.getStreamSize(0));
  std::vector<uint32_t> NewStreamBlocks = Msf.getStreamBlocks(0);
  EXPECT_EQ(1U, NewStreamBlocks.size());

  EXPECT_EQ(OldStreamBlocks[0], NewStreamBlocks[0]);
}

TEST_F(MSFBuilderTest, TestRejectReusedStreamBlock) {
  // Test that attempting to add a stream and assigning a block that is already
  // in use by another stream fails.
  auto ExpectedMsf = MSFBuilder::create(Allocator, 4096);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  EXPECT_THAT_EXPECTED(Msf.addStream(6144), Succeeded());

  std::vector<uint32_t> Blocks = {2, 3};
  EXPECT_THAT_EXPECTED(Msf.addStream(6144, Blocks), Failed());
}

TEST_F(MSFBuilderTest, TestBlockCountsWhenAddingStreams) {
  // Test that when adding multiple streams, the number of used and free Blocks
  // allocated to the MSF file are as expected.
  auto ExpectedMsf = MSFBuilder::create(Allocator, 4096);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  // one for the super block, one for the directory block map
  uint32_t NumUsedBlocks = Msf.getNumUsedBlocks();
  EXPECT_EQ(msf::getMinimumBlockCount(), NumUsedBlocks);
  EXPECT_EQ(0U, Msf.getNumFreeBlocks());

  const uint32_t StreamSizes[] = {4000, 6193, 189723};
  for (int I = 0; I < 3; ++I) {
    EXPECT_THAT_EXPECTED(Msf.addStream(StreamSizes[I]), Succeeded());
    NumUsedBlocks += bytesToBlocks(StreamSizes[I], 4096);
    EXPECT_EQ(NumUsedBlocks, Msf.getNumUsedBlocks());
    EXPECT_EQ(0U, Msf.getNumFreeBlocks());
  }
}

TEST_F(MSFBuilderTest, BuildMsfLayout) {
  // Test that we can generate an MSFLayout structure from a valid layout
  // specification.
  auto ExpectedMsf = MSFBuilder::create(Allocator, 4096);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  const uint32_t StreamSizes[] = {4000, 6193, 189723};
  uint32_t ExpectedNumBlocks = msf::getMinimumBlockCount();
  for (int I = 0; I < 3; ++I) {
    EXPECT_THAT_EXPECTED(Msf.addStream(StreamSizes[I]), Succeeded());
    ExpectedNumBlocks += bytesToBlocks(StreamSizes[I], 4096);
  }
  ++ExpectedNumBlocks; // The directory itself should use 1 block

  auto ExpectedLayout = Msf.generateLayout();
  EXPECT_THAT_EXPECTED(ExpectedLayout, Succeeded());
  MSFLayout &L = *ExpectedLayout;
  EXPECT_EQ(4096U, L.SB->BlockSize);
  EXPECT_EQ(ExpectedNumBlocks, L.SB->NumBlocks);

  EXPECT_EQ(1U, L.DirectoryBlocks.size());

  EXPECT_EQ(3U, L.StreamMap.size());
  EXPECT_EQ(3U, L.StreamSizes.size());
  for (int I = 0; I < 3; ++I) {
    EXPECT_EQ(StreamSizes[I], L.StreamSizes[I]);
    uint32_t ExpectedNumBlocks = bytesToBlocks(StreamSizes[I], 4096);
    EXPECT_EQ(ExpectedNumBlocks, L.StreamMap[I].size());
  }
}

TEST_F(MSFBuilderTest, UseDirectoryBlockHint) {
  Expected<MSFBuilder> ExpectedMsf = MSFBuilder::create(
      Allocator, 4096, msf::getMinimumBlockCount() + 1, false);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  uint32_t B = msf::getFirstUnreservedBlock();
  EXPECT_THAT_ERROR(Msf.setDirectoryBlocksHint({B + 1}), Succeeded());
  EXPECT_THAT_EXPECTED(Msf.addStream(2048, {B + 2}), Succeeded());

  auto ExpectedLayout = Msf.generateLayout();
  EXPECT_THAT_EXPECTED(ExpectedLayout, Succeeded());
  MSFLayout &L = *ExpectedLayout;
  EXPECT_EQ(msf::getMinimumBlockCount() + 2, L.SB->NumBlocks);
  EXPECT_EQ(1U, L.DirectoryBlocks.size());
  EXPECT_EQ(1U, L.StreamMap[0].size());

  EXPECT_EQ(B + 1, L.DirectoryBlocks[0]);
  EXPECT_EQ(B + 2, L.StreamMap[0].front());
}

TEST_F(MSFBuilderTest, DirectoryBlockHintInsufficient) {
  Expected<MSFBuilder> ExpectedMsf =
      MSFBuilder::create(Allocator, 4096, msf::getMinimumBlockCount() + 2);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;
  uint32_t B = msf::getFirstUnreservedBlock();
  EXPECT_THAT_ERROR(Msf.setDirectoryBlocksHint({B + 1}), Succeeded());

  uint32_t Size = 4096 * 4096 / 4;
  EXPECT_THAT_EXPECTED(Msf.addStream(Size), Succeeded());

  auto ExpectedLayout = Msf.generateLayout();
  EXPECT_THAT_EXPECTED(ExpectedLayout, Succeeded());
  MSFLayout &L = *ExpectedLayout;
  EXPECT_EQ(2U, L.DirectoryBlocks.size());
  EXPECT_EQ(B + 1, L.DirectoryBlocks[0]);
}

TEST_F(MSFBuilderTest, DirectoryBlockHintOverestimated) {
  Expected<MSFBuilder> ExpectedMsf =
      MSFBuilder::create(Allocator, 4096, msf::getMinimumBlockCount() + 2);
  EXPECT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  uint32_t B = msf::getFirstUnreservedBlock();
  EXPECT_THAT_ERROR(Msf.setDirectoryBlocksHint({B + 1, B + 2}), Succeeded());

  ASSERT_THAT_EXPECTED(Msf.addStream(2048), Succeeded());

  auto ExpectedLayout = Msf.generateLayout();
  ASSERT_THAT_EXPECTED(ExpectedLayout, Succeeded());
  MSFLayout &L = *ExpectedLayout;
  EXPECT_EQ(1U, L.DirectoryBlocks.size());
  EXPECT_EQ(B + 1, L.DirectoryBlocks[0]);
}

TEST_F(MSFBuilderTest, StreamDoesntUseFpmBlocks) {
  Expected<MSFBuilder> ExpectedMsf = MSFBuilder::create(Allocator, 4096);
  ASSERT_THAT_EXPECTED(ExpectedMsf, Succeeded());
  auto &Msf = *ExpectedMsf;

  // A block is 4096 bytes, and every 4096 blocks we have 2 reserved FPM blocks.
  // By creating add a stream that spans 4096*4096*3 bytes, we ensure that we
  // cross over a couple of reserved FPM blocks, and that none of them are
  // allocated to the stream.
  constexpr uint32_t StreamSize = 4096 * 4096 * 3;
  Expected<uint32_t> SN = Msf.addStream(StreamSize);
  ASSERT_THAT_EXPECTED(SN, Succeeded());

  auto ExpectedLayout = Msf.generateLayout();
  ASSERT_THAT_EXPECTED(ExpectedLayout, Succeeded());
  MSFLayout &L = *ExpectedLayout;
  auto BlocksRef = L.StreamMap[*SN];
  std::vector<uint32_t> Blocks(BlocksRef.begin(), BlocksRef.end());
  EXPECT_EQ(StreamSize, L.StreamSizes[*SN]);

  for (uint32_t I = 0; I <= 3; ++I) {
    // Pages from both FPMs are always allocated.
    EXPECT_FALSE(L.FreePageMap.test(2 + I * 4096));
    EXPECT_FALSE(L.FreePageMap.test(1 + I * 4096));
  }

  for (uint32_t I = 1; I <= 3; ++I) {
    EXPECT_THAT(Blocks, Not(Contains(1 + I * 4096)));
    EXPECT_THAT(Blocks, Not(Contains(2 + I * 4096)));
  }
}
