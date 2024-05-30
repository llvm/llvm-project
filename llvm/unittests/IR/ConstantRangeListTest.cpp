//===- ConstantRangeListTest.cpp - ConstantRangeList tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantRangeList.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

using ConstantRangeListTest = ::testing::Test;

TEST_F(ConstantRangeListTest, Basics) {
  ConstantRangeList CRL1a;
  CRL1a.insert(0, 12);
  EXPECT_FALSE(CRL1a.empty());

  ConstantRangeList CRL1b;
  CRL1b.insert(0, 4);
  CRL1b.insert(4, 8);
  CRL1b.insert(8, 12);
  EXPECT_TRUE(CRL1a == CRL1b);

  ConstantRangeList CRL1c;
  CRL1c.insert(0, 4);
  CRL1c.insert(8, 12);
  CRL1c.insert(4, 8);
  EXPECT_TRUE(CRL1a == CRL1c);

  ConstantRangeList CRL2;
  CRL2.insert(-4, 0);
  CRL2.insert(8, 12);
  EXPECT_TRUE(CRL1a != CRL2);
}

TEST_F(ConstantRangeListTest, getConstantRangeList) {
  SmallVector<ConstantRange, 2> Empty;
  EXPECT_TRUE(ConstantRangeList::getConstantRangeList(Empty).has_value());

  SmallVector<ConstantRange, 2> Valid;
  Valid.push_back(ConstantRange(APInt(64, 0, true), APInt(64, 4, true)));
  Valid.push_back(ConstantRange(APInt(64, 8, true), APInt(64, 12, true)));
  EXPECT_TRUE(ConstantRangeList::getConstantRangeList(Valid).has_value());

  SmallVector<ConstantRange, 2> Invalid1;
  Invalid1.push_back(ConstantRange(APInt(64, 4, true), APInt(64, 0, true)));
  EXPECT_EQ(ConstantRangeList::getConstantRangeList(Invalid1), std::nullopt);

  SmallVector<ConstantRange, 2> Invalid2;
  Invalid2.push_back(ConstantRange(APInt(64, 0, true), APInt(64, 4, true)));
  Invalid2.push_back(ConstantRange(APInt(64, 12, true), APInt(64, 8, true)));
  EXPECT_EQ(ConstantRangeList::getConstantRangeList(Invalid2), std::nullopt);

  SmallVector<ConstantRange, 2> Invalid3;
  Invalid3.push_back(ConstantRange(APInt(64, 0, true), APInt(64, 4, true)));
  Invalid3.push_back(ConstantRange(APInt(64, 4, true), APInt(64, 8, true)));
  EXPECT_EQ(ConstantRangeList::getConstantRangeList(Invalid3), std::nullopt);

  SmallVector<ConstantRange, 2> Invalid4;
  Invalid4.push_back(ConstantRange(APInt(64, 0, true), APInt(64, 12, true)));
  Invalid4.push_back(ConstantRange(APInt(64, 8, true), APInt(64, 16, true)));
  EXPECT_EQ(ConstantRangeList::getConstantRangeList(Invalid4), std::nullopt);
}

TEST_F(ConstantRangeListTest, Insert) {
  ConstantRangeList CRL;
  CRL.insert(0, 4);
  CRL.insert(8, 12);
  // No overlap, left
  CRL.insert(-8, -4);
  // No overlap, right
  CRL.insert(16, 20);
  // No overlap, middle
  CRL.insert(13, 15);
  // Overlap with left
  CRL.insert(-6, -2);
  // Overlap with right
  CRL.insert(5, 9);
  // Overlap with left and right
  CRL.insert(14, 18);
  // Overlap cross ranges
  CRL.insert(2, 14);
  // An existing range
  CRL.insert(0, 20);

  ConstantRangeList Expected;
  Expected.insert(-8, -2);
  Expected.insert(0, 20);
  EXPECT_TRUE(CRL == Expected);
}

} // anonymous namespace
