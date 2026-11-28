//===- HexagonRelocationTest.cpp - Hexagon relocation API tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/Hexagon/HexagonRelocation.h"
#include "llvm/BinaryFormat/ELF.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(HexagonRelocationTest, GetRelocationInfo) {
  LLVMHexagonRelocInfo Info;
  ASSERT_TRUE(LLVMHexagonGetRelocationInfo(ELF::R_HEX_B22_PCREL, &Info));
  EXPECT_STREQ("R_HEX_B22_PCREL", Info.Name);
  EXPECT_EQ(uint32_t(ELF::R_HEX_B22_PCREL), Info.Type);
  EXPECT_EQ(22u, Info.EffectiveBits);
  EXPECT_EQ(4u, Info.Alignment);
  EXPECT_EQ(2u, Info.Shift);
  EXPECT_TRUE(Info.VerifyRange);
  EXPECT_TRUE(Info.VerifyAlignment);
  EXPECT_TRUE(Info.IsSigned);

  // The last known relocation is accepted; anything past it is rejected and
  // leaves Info untouched.
  EXPECT_TRUE(LLVMHexagonGetRelocationInfo(ELF::R_HEX_27_REG, &Info));
  Info.Type = 0xdead;
  EXPECT_FALSE(LLVMHexagonGetRelocationInfo(ELF::R_HEX_27_REG + 1, &Info));
  EXPECT_EQ(0xdeadu, Info.Type);
}

TEST(HexagonRelocationTest, VerifyRange) {
  // Signed, 22 bits.
  EXPECT_TRUE(LLVMHexagonVerifyRange(ELF::R_HEX_B22_PCREL, 0x1fffff));
  EXPECT_FALSE(LLVMHexagonVerifyRange(ELF::R_HEX_B22_PCREL, 0x200000));
  EXPECT_TRUE(LLVMHexagonVerifyRange(ELF::R_HEX_B22_PCREL, 0xffe00000u));
  EXPECT_FALSE(LLVMHexagonVerifyRange(ELF::R_HEX_B22_PCREL, 0xffdfffffu));
  // Unsigned, 8 bits.
  EXPECT_TRUE(LLVMHexagonVerifyRange(ELF::R_HEX_8, 0xff));
  EXPECT_FALSE(LLVMHexagonVerifyRange(ELF::R_HEX_8, 0x100));
  // Unknown type.
  EXPECT_FALSE(LLVMHexagonVerifyRange(ELF::R_HEX_27_REG + 1, 0));
}

TEST(HexagonRelocationTest, VerifyAlignment) {
  EXPECT_TRUE(LLVMHexagonVerifyAlignment(ELF::R_HEX_B22_PCREL, 8));
  EXPECT_FALSE(LLVMHexagonVerifyAlignment(ELF::R_HEX_B22_PCREL, 6));
  EXPECT_TRUE(LLVMHexagonVerifyAlignment(ELF::R_HEX_8, 3));
  EXPECT_FALSE(LLVMHexagonVerifyAlignment(ELF::R_HEX_27_REG + 1, 0));
}

TEST(HexagonRelocationTest, IsTruncated) {
  EXPECT_TRUE(LLVMHexagonIsTruncated(ELF::R_HEX_8, 0x100));
  EXPECT_FALSE(LLVMHexagonIsTruncated(ELF::R_HEX_8, 0xff));
  // Signed and unknown relocations are never reported as truncated.
  EXPECT_FALSE(LLVMHexagonIsTruncated(ELF::R_HEX_B22_PCREL, 0x7fffffff));
  EXPECT_FALSE(LLVMHexagonIsTruncated(ELF::R_HEX_27_REG + 1, 0xffffffff));
}

TEST(HexagonRelocationTest, ApplyRelocation) {
  EXPECT_EQ(0x12345678u,
            LLVMHexagonApplyRelocation(ELF::R_HEX_32, 0, 0x12345678));
  // Apply takes the raw value and shifts internally (B22_PCREL: >> 2), whereas
  // VerifyRange takes the already shifted value.
  EXPECT_EQ(0x4u, LLVMHexagonApplyRelocation(ELF::R_HEX_B22_PCREL, 0, 8));
  // Unknown relocation types leave the instruction alone.
  EXPECT_EQ(0xdeadbeefu,
            LLVMHexagonApplyRelocation(ELF::R_HEX_27_REG + 1, 0xdeadbeef, 1));
}
