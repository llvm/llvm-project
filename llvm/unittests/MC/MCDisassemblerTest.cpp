//===- MCDisassemblerTest.cpp - Tests for MCDisassembler.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(MCDisassembler, XCOFFSymbolPriorityTest) {

  SymbolInfoTy SIT1(0x100000, "sym1", None, 1, false);
  SymbolInfoTy SIT2(0x110000, "sym2", None, 2, false);
  SymbolInfoTy SIT3(0x120000, ".func", XCOFF::XMC_PR, 3, true);
  SymbolInfoTy SIT4(0x120000, ".text", XCOFF::XMC_PR, 4, false);
  SymbolInfoTy SIT5(0x130000, "TOC", XCOFF::XMC_TC0, 5, false);
  SymbolInfoTy SIT6(0x130000, "func", XCOFF::XMC_TC, 6, false);

  // Test that higher addresses would appear later than lower ones when symbols
  // are sorted in ascending order.
  EXPECT_TRUE(SIT1 < SIT2);
  EXPECT_FALSE(SIT2 < SIT1);

  // Test that symbols with a StorageMappingClass have higher priority than those
  // without.
  EXPECT_TRUE(SIT2 < SIT5);
  EXPECT_FALSE(SIT5 < SIT2);

  // Test that symbols with a TC0 StorageMappingClass have lower priority than those
  // with some other StorageMappingClass.
  EXPECT_TRUE(SIT5 < SIT6);
  EXPECT_FALSE(SIT6 < SIT5);

  // Test label symbols have higher priorty than non-label symbols.
  EXPECT_TRUE(SIT4 < SIT3);
  EXPECT_FALSE(SIT3 < SIT4);

  // Test symbols comparing with themselves.
  EXPECT_FALSE(SIT1 < SIT1);
  EXPECT_FALSE(SIT2 < SIT2);
  EXPECT_FALSE(SIT3 < SIT3);
  EXPECT_FALSE(SIT4 < SIT4);
  EXPECT_FALSE(SIT5 < SIT5);
  EXPECT_FALSE(SIT6 < SIT6);
}
