//===- COFFObjectFileTest.cpp - Tests for COFFObjectFile ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/COFF.h"
#include "gtest/gtest.h"

using namespace llvm::object;

TEST(COFFObjectFileTest, CHPERangeEntry) {
  chpe_range_entry range;

  range.StartOffset = 0x1000;
  EXPECT_EQ(range.getStart(), 0x1000u);
  EXPECT_EQ(range.getType(), chpe_range_type::Arm64);

  range.StartOffset = 0x2000 | chpe_range_type::Arm64EC;
  EXPECT_EQ(range.getStart(), 0x2000u);
  EXPECT_EQ(range.getType(), chpe_range_type::Arm64EC);

  range.StartOffset = 0x3000 | chpe_range_type::Amd64;
  EXPECT_EQ(range.getStart(), 0x3000u);
  EXPECT_EQ(range.getType(), chpe_range_type::Amd64);
}
