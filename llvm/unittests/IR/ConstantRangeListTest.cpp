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
  CRL1.append(0, 4);
  CRL1.append(8, 12);
  EXPECT_FALSE(CRL1.isFullSet());
  EXPECT_FALSE(CRL1.isEmptySet());

  ConstantRangeList CRL2(64, false);
  CRL2.append(0, 4);
  CRL2.append(8, 12);
  EXPECT_TRUE(CRL1 == CRL2);

  ConstantRangeList CRL3(64, false);
  CRL3.append(-4, 0);
  CRL3.append(8, 12);
  EXPECT_TRUE(CRL1 != CRL3);
}

} // anonymous namespace
