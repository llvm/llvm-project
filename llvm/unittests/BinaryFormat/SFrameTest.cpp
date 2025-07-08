//===- SFrameTest.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/SFrame.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sframe;

namespace {
// Test structure sizes and triviality.
static_assert(std::is_trivial_v<Preamble>);
static_assert(sizeof(Preamble) == 4);

static_assert(std::is_trivial_v<Header>);
static_assert(sizeof(Header) == 28);

static_assert(std::is_trivial_v<FuncDescEntry>);
static_assert(sizeof(FuncDescEntry) == 20);

static_assert(std::is_trivial_v<FrameRowEntryAddr1>);
static_assert(sizeof(FrameRowEntryAddr1) == 2);

static_assert(std::is_trivial_v<FrameRowEntryAddr2>);
static_assert(sizeof(FrameRowEntryAddr2) == 3);

static_assert(std::is_trivial_v<FrameRowEntryAddr4>);
static_assert(sizeof(FrameRowEntryAddr4) == 5);

TEST(SFrameTest, FDEFlags) {
  FuncDescEntry FDE = {};
  EXPECT_EQ(FDE.Info, 0u);
  EXPECT_EQ(FDE.getPAuthKey(), 0);
  EXPECT_EQ(FDE.getFDEType(), FDEType::PCInc);
  EXPECT_EQ(FDE.getFREType(), FREType::Addr1);

  FDE.setPAuthKey(1);
  EXPECT_EQ(FDE.Info, 0x20u);
  EXPECT_EQ(FDE.getPAuthKey(), 1);
  EXPECT_EQ(FDE.getFDEType(), FDEType::PCInc);
  EXPECT_EQ(FDE.getFREType(), FREType::Addr1);

  FDE.setFDEType(FDEType::PCMask);
  EXPECT_EQ(FDE.Info, 0x30u);
  EXPECT_EQ(FDE.getPAuthKey(), 1);
  EXPECT_EQ(FDE.getFDEType(), FDEType::PCMask);
  EXPECT_EQ(FDE.getFREType(), FREType::Addr1);

  FDE.setFREType(FREType::Addr4);
  EXPECT_EQ(FDE.Info, 0x32u);
  EXPECT_EQ(FDE.getPAuthKey(), 1);
  EXPECT_EQ(FDE.getFDEType(), FDEType::PCMask);
  EXPECT_EQ(FDE.getFREType(), FREType::Addr4);
}

TEST(SFrameTest, FREFlags) {
  FREInfo Info = {};
  EXPECT_EQ(Info.Info, 0u);
  EXPECT_FALSE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), FREOffset::B1);
  EXPECT_EQ(Info.getOffsetCount(), 0u);
  EXPECT_EQ(Info.getBaseRegister(), BaseReg::FP);

  Info.setReturnAddressSigned(true);
  EXPECT_EQ(Info.Info, 0x80u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), FREOffset::B1);
  EXPECT_EQ(Info.getOffsetCount(), 0u);
  EXPECT_EQ(Info.getBaseRegister(), BaseReg::FP);

  Info.setOffsetSize(FREOffset::B4);
  EXPECT_EQ(Info.Info, 0xc0u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), FREOffset::B4);
  EXPECT_EQ(Info.getOffsetCount(), 0u);
  EXPECT_EQ(Info.getBaseRegister(), BaseReg::FP);

  Info.setOffsetCount(3);
  EXPECT_EQ(Info.Info, 0xc6u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), FREOffset::B4);
  EXPECT_EQ(Info.getOffsetCount(), 3u);
  EXPECT_EQ(Info.getBaseRegister(), BaseReg::FP);

  Info.setBaseRegister(BaseReg::SP);
  EXPECT_EQ(Info.Info, 0xc7u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), FREOffset::B4);
  EXPECT_EQ(Info.getOffsetCount(), 3u);
  EXPECT_EQ(Info.getBaseRegister(), BaseReg::SP);
}

} // namespace
