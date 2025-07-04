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
static_assert(std::is_trivial_v<sframe_preamble>);
static_assert(sizeof(sframe_preamble) == 4);

static_assert(std::is_trivial_v<sframe_header>);
static_assert(sizeof(sframe_header) == 28);

static_assert(std::is_trivial_v<sframe_func_desc_entry>);
static_assert(sizeof(sframe_func_desc_entry) == 20);

static_assert(std::is_trivial_v<sframe_frame_row_entry_addr1>);
static_assert(sizeof(sframe_frame_row_entry_addr1) == 2);

static_assert(std::is_trivial_v<sframe_frame_row_entry_addr2>);
static_assert(sizeof(sframe_frame_row_entry_addr2) == 3);

static_assert(std::is_trivial_v<sframe_frame_row_entry_addr4>);
static_assert(sizeof(sframe_frame_row_entry_addr4) == 5);

TEST(SFrameTest, FDEFlags) {
  sframe_func_desc_entry FDE = {};
  EXPECT_EQ(FDE.sfde_func_info, 0u);
  EXPECT_EQ(FDE.getPAuthKey(), SFRAME_AARCH64_PAUTH_KEY_A);
  EXPECT_EQ(FDE.getFDEType(), SFRAME_FDE_TYPE_PCINC);
  EXPECT_EQ(FDE.getFREType(), SFRAME_FRE_TYPE_ADDR1);

  FDE.setPAuthKey(SFRAME_AARCH64_PAUTH_KEY_B);
  EXPECT_EQ(FDE.sfde_func_info, 0x20u);
  EXPECT_EQ(FDE.getPAuthKey(), SFRAME_AARCH64_PAUTH_KEY_B);
  EXPECT_EQ(FDE.getFDEType(), SFRAME_FDE_TYPE_PCINC);
  EXPECT_EQ(FDE.getFREType(), SFRAME_FRE_TYPE_ADDR1);

  FDE.setFDEType(SFRAME_FDE_TYPE_PCMASK);
  EXPECT_EQ(FDE.sfde_func_info, 0x30u);
  EXPECT_EQ(FDE.getPAuthKey(), SFRAME_AARCH64_PAUTH_KEY_B);
  EXPECT_EQ(FDE.getFDEType(), SFRAME_FDE_TYPE_PCMASK);
  EXPECT_EQ(FDE.getFREType(), SFRAME_FRE_TYPE_ADDR1);

  FDE.setFREType(SFRAME_FRE_TYPE_ADDR4);
  EXPECT_EQ(FDE.sfde_func_info, 0x32u);
  EXPECT_EQ(FDE.getPAuthKey(), SFRAME_AARCH64_PAUTH_KEY_B);
  EXPECT_EQ(FDE.getFDEType(), SFRAME_FDE_TYPE_PCMASK);
  EXPECT_EQ(FDE.getFREType(), SFRAME_FRE_TYPE_ADDR4);
}

TEST(SFrameTest, FREFlags) {
  sframe_fre_info Info = {};
  EXPECT_EQ(Info.fre_info, 0u);
  EXPECT_FALSE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), SFRAME_FRE_OFFSET_1B);
  EXPECT_EQ(Info.getOffsetCount(), 0u);
  EXPECT_EQ(Info.getBaseRegister(), SFRAME_BASE_REG_FP);

  Info.setReturnAddressSigned(true);
  EXPECT_EQ(Info.fre_info, 0x80u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), SFRAME_FRE_OFFSET_1B);
  EXPECT_EQ(Info.getOffsetCount(), 0u);
  EXPECT_EQ(Info.getBaseRegister(), SFRAME_BASE_REG_FP);

  Info.setOffsetSize(SFRAME_FRE_OFFSET_4B);
  EXPECT_EQ(Info.fre_info, 0xc0u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), SFRAME_FRE_OFFSET_4B);
  EXPECT_EQ(Info.getOffsetCount(), 0u);
  EXPECT_EQ(Info.getBaseRegister(), SFRAME_BASE_REG_FP);

  Info.setOffsetCount(3);
  EXPECT_EQ(Info.fre_info, 0xc6u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), SFRAME_FRE_OFFSET_4B);
  EXPECT_EQ(Info.getOffsetCount(), 3u);
  EXPECT_EQ(Info.getBaseRegister(), SFRAME_BASE_REG_FP);

  Info.setBaseRegister(SFRAME_BASE_REG_SP);
  EXPECT_EQ(Info.fre_info, 0xc7u);
  EXPECT_TRUE(Info.isReturnAddressSigned());
  EXPECT_EQ(Info.getOffsetSize(), SFRAME_FRE_OFFSET_4B);
  EXPECT_EQ(Info.getOffsetCount(), 3u);
  EXPECT_EQ(Info.getBaseRegister(), SFRAME_BASE_REG_SP);
}

} // namespace
