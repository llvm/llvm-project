//===- MCInstrDescTest.cpp - MCInstrDesc unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstrDesc.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(MCInstrDescTest, PackedFields) {
  struct OperandCounts {
    unsigned NumOperands;
    unsigned NumDefs;
  };

  constexpr OperandCounts TestCounts[] = {
      {0, 0},   {1, 0},     {1, 1},    {33, 8},    {67, 65},
      {130, 0}, {131, 129}, {255, 31}, {255, 224}, {255, 255}};
  for (OperandCounts Counts : TestCounts) {
    for (unsigned Size : {0U, 2U, 3U, 4U, 63U, 64U, 252U, 255U}) {
      constexpr uint64_t Flags = (1ULL << MCID::Authenticated) | 1;
      MCInstrDesc Desc(
          MCInstrDesc::TableGenEncoding{}, UINT64_MAX,
          MCInstrDesc::TableGenEncoding::encodeFlagsAndImplicit(Flags, Size, 0),
          MCInstrDesc::TableGenEncoding::encodeOpcodeAndOperands(
              65535, Counts.NumOperands, Counts.NumDefs, 65535, 65535));

      EXPECT_EQ(Desc.getOpcode(), 65535U);
      EXPECT_EQ(Desc.getNumOperands(), Counts.NumOperands);
      EXPECT_EQ(Desc.getNumDefs(), Counts.NumDefs);
      EXPECT_EQ(Desc.getSize(), Size);
      EXPECT_EQ(Desc.getSchedClass(), 65535U);
      EXPECT_TRUE(Desc.isPreISelOpcode());
      EXPECT_TRUE(Desc.isAuthenticated());
      EXPECT_EQ(Desc.TSFlags, UINT64_MAX);
    }
  }
}

TEST(MCInstrDescTest, EmptyImplicitOperands) {
  MCInstrDesc Desc;
  EXPECT_EQ(Desc.getNumImplicitUses(), 0U);
  EXPECT_EQ(Desc.getNumImplicitDefs(), 0U);
  EXPECT_TRUE(Desc.implicit_uses().empty());
  EXPECT_TRUE(Desc.implicit_defs().empty());
}

TEST(MCInstrDescTest, ImplicitOperands) {
  constexpr unsigned NumImplicitUses = 2;
  constexpr unsigned NumImplicitDefs = 3;
  constexpr MCPhysReg Header =
      MCInstrDesc::TableGenEncoding::encodeImplicitHeader(NumImplicitUses,
                                                          NumImplicitDefs);
  struct InstrTable {
    MCInstrDesc Insts[1];
    MCPhysReg ImplicitOps[2 + NumImplicitUses + NumImplicitDefs];
  } Table{{MCInstrDesc(
              MCInstrDesc::TableGenEncoding{}, 0,
              MCInstrDesc::TableGenEncoding::encodeFlagsAndImplicit(0, 0, 1),
              MCInstrDesc::TableGenEncoding::encodeOpcodeAndOperands(0, 0, 0, 0,
                                                                     0))},
          {0, Header, 11, 12, 13, 14, 15}};

  const MCInstrDesc &Desc = Table.Insts[0];
  EXPECT_EQ(Desc.getNumImplicitUses(), NumImplicitUses);
  EXPECT_EQ(Desc.getNumImplicitDefs(), NumImplicitDefs);
  ASSERT_EQ(Desc.implicit_uses().size(), NumImplicitUses);
  EXPECT_EQ(Desc.implicit_uses()[0], 11);
  EXPECT_EQ(Desc.implicit_uses()[1], 12);
  ASSERT_EQ(Desc.implicit_defs().size(), NumImplicitDefs);
  EXPECT_EQ(Desc.implicit_defs()[0], 13);
  EXPECT_EQ(Desc.implicit_defs()[1], 14);
  EXPECT_EQ(Desc.implicit_defs()[2], 15);
}

} // namespace
