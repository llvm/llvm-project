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

class ConstantRangeListTest : public ::testing::Test {
protected:
  static ConstantRangeList Full;
  static ConstantRangeList Empty;
};

ConstantRangeList ConstantRangeListTest::Full(64, true);
ConstantRangeList ConstantRangeListTest::Empty(64, false);

TEST_F(ConstantRangeListTest, Basics) {
  EXPECT_TRUE(Full.isFullSet());
  EXPECT_FALSE(Full.isEmptySet());

  EXPECT_FALSE(Empty.isFullSet());
  EXPECT_TRUE(Empty.isEmptySet());

  ConstantRangeList CRL1(64, false);
  CRL1.insert(0, 4);
  CRL1.insert(8, 12);
  EXPECT_FALSE(CRL1.isFullSet());
  EXPECT_FALSE(CRL1.isEmptySet());

  ConstantRangeList CRL2(64, false);
  CRL2.insert(0, 4);
  CRL2.insert(8, 12);
  EXPECT_TRUE(CRL1 == CRL2);

  ConstantRangeList CRL3(64, false);
  CRL3.insert(-4, 0);
  CRL3.insert(8, 12);
  EXPECT_TRUE(CRL1 != CRL3);
}

TEST_F(ConstantRangeListTest, Insert) {
  ConstantRangeList CRL(64, false);
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

  ConstantRangeList Expected(64, false);
  Expected.insert(-8, -2);
  Expected.insert(0, 20);
  EXPECT_TRUE(CRL == Expected);
}

} // anonymous namespace
