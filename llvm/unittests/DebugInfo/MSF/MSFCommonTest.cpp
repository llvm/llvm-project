//===- MSFBuilderTest.cpp  Tests manipulation of MSF stream metadata ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::msf;

TEST(MSFCommonTest, BytesToBlocks) {
  EXPECT_EQ(0ULL, bytesToBlocks(0, 4096));
  EXPECT_EQ(1ULL, bytesToBlocks(12, 4096));
  EXPECT_EQ(1ULL, bytesToBlocks(4096, 4096));
  EXPECT_EQ(2ULL, bytesToBlocks(4097, 4096));
  EXPECT_EQ(2ULL, bytesToBlocks(600, 512));
}

TEST(MSFCommonTest, FpmIntervals) {
  SuperBlock SB;
  SB.FreeBlockMapBlock = 1;
  SB.BlockSize = 4096;

  MSFLayout L;
  L.SB = &SB;

  SB.NumBlocks = 12;
  EXPECT_EQ(1u, getNumFpmIntervals(L, false));
  SB.NumBlocks = SB.BlockSize;
  EXPECT_EQ(1u, getNumFpmIntervals(L, false));
  SB.NumBlocks = SB.BlockSize + 1;
  EXPECT_EQ(1u, getNumFpmIntervals(L, false));
  SB.NumBlocks = SB.BlockSize * 8;
  EXPECT_EQ(1u, getNumFpmIntervals(L, false));
  SB.NumBlocks = SB.BlockSize * 8 + 1;
  EXPECT_EQ(2u, getNumFpmIntervals(L, false));

  SB.NumBlocks = 12;
  EXPECT_EQ(1u, getNumFpmIntervals(L, true));
  SB.NumBlocks = SB.BlockSize;
  EXPECT_EQ(1u, getNumFpmIntervals(L, true));
  SB.NumBlocks = SB.BlockSize + 1;
  EXPECT_EQ(2u, getNumFpmIntervals(L, true));
  SB.NumBlocks = SB.BlockSize * 8;
  EXPECT_EQ(8u, getNumFpmIntervals(L, true));
  SB.NumBlocks = SB.BlockSize * 8 + 1;
  EXPECT_EQ(9u, getNumFpmIntervals(L, true));
}

TEST(MSFCommonTest, FpmStreamLayout) {
  SuperBlock SB;
  MSFLayout L;
  L.SB = &SB;
  SB.FreeBlockMapBlock = 1;

  // Each FPM block has 4096 bytes for a maximum of 4096*8 allocation states.
  SB.BlockSize = 4096;

  // 1. When we're not including unused FPM data, the length of the FPM stream
  //    should be only long enough to contain 1 bit for each block.

  // 1a. When the PDB has <= 4096*8 blocks, there should only be one FPM block.
  SB.NumBlocks = 8000;
  MSFStreamLayout SL = getFpmStreamLayout(L, false, false);
  EXPECT_EQ(1000u, SL.Length);
  EXPECT_EQ(1u, SL.Blocks.size());
  EXPECT_EQ(SB.FreeBlockMapBlock, SL.Blocks.front());

  SL = getFpmStreamLayout(L, false, true);
  EXPECT_EQ(1000u, SL.Length);
  EXPECT_EQ(1u, SL.Blocks.size());
  EXPECT_EQ(3u - SB.FreeBlockMapBlock, SL.Blocks.front());

  // 1b. When the PDB has > 4096*8 blocks, there should be multiple FPM blocks.
  SB.NumBlocks = SB.BlockSize * 8 + 1;
  SL = getFpmStreamLayout(L, false, false);
  EXPECT_EQ(SB.BlockSize + 1, SL.Length);
  EXPECT_EQ(2u, SL.Blocks.size());
  EXPECT_EQ(SB.FreeBlockMapBlock, SL.Blocks[0]);
  EXPECT_EQ(SB.FreeBlockMapBlock + SB.BlockSize, SL.Blocks[1]);

  SL = getFpmStreamLayout(L, false, true);
  EXPECT_EQ(SB.BlockSize + 1, SL.Length);
  EXPECT_EQ(2u, SL.Blocks.size());
  EXPECT_EQ(3u - SB.FreeBlockMapBlock, SL.Blocks[0]);
  EXPECT_EQ(3u - SB.FreeBlockMapBlock + SB.BlockSize, SL.Blocks[1]);

  // 2. When we are including unused FPM data, there should be one FPM block
  //    at every BlockSize interval in the file, even if entire FPM blocks are
  //    unused.
  SB.NumBlocks = SB.BlockSize * 8 + 1;
  SL = getFpmStreamLayout(L, true, false);
  EXPECT_EQ(SB.BlockSize * 9, SL.Length);
  EXPECT_EQ(9u, SL.Blocks.size());
  for (int I = 0; I < 9; ++I)
    EXPECT_EQ(I * SB.BlockSize + SB.FreeBlockMapBlock, SL.Blocks[I]);
}
