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
  ConstantRangeList CRL1;
  CRL1.insert(0, 4);
  CRL1.insert(8, 12);
  EXPECT_FALSE(CRL1.empty());

  ConstantRangeList CRL2;
  CRL2.insert(0, 4);
  CRL2.insert(8, 12);
  EXPECT_TRUE(CRL1 == CRL2);

  ConstantRangeList CRL3;
  CRL3.insert(-4, 0);
  CRL3.insert(8, 12);
  EXPECT_TRUE(CRL1 != CRL3);
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
