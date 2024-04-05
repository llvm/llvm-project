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
public:
  ConstantRangeList
  GetCRL(SmallVectorImpl<std::pair<int64_t, int64_t>> &Ranges) {
    ConstantRangeList Result(64, false);
    for (const auto &[Start, End] : Ranges) {
      Result.append(Start, End);
    }
    return Result;
  }

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

  ConstantRangeList Some1 = GetCRL({(0, 4), (8, 12)});
  EXPECT_FALSE(Some1.isFullSet());
  EXPECT_FALSE(Some1.isEmptySet());

  ConstantRangeList Some2 = GetCRL({(0, 4), (8, 12)});
  ConstantRangeList Some3 = GetCRL({(-4, 0), (8, 12)});
  EXPECT_TRUE(Some1 == Some2 && Some1 != Some3);
}

} // anonymous namespace
