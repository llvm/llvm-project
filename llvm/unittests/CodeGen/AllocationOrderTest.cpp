//===- llvm/unittest/CodeGen/AllocationOrderTest.cpp - AllocationOrder tests =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/CodeGen/AllocationOrder.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
std::vector<MCPhysReg> loadOrder(AllocationOrder &O, unsigned Limit = 0) {
  std::vector<MCPhysReg> Ret;
  O.rewind();
  while (auto R = O.next(Limit))
    Ret.push_back(R);
  return Ret;
}
} // namespace

TEST(AllocationOrderTest, Basic) {
  SmallVector<MCPhysReg, 16> Hints = {1, 2, 3};
  SmallVector<MCPhysReg, 16> Order = {4, 5, 6, 7};
  AllocationOrder O(std::move(Hints), Order, false);
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4, 5, 6, 7}), loadOrder(O));
}

TEST(AllocationOrderTest, Duplicates) {
  SmallVector<MCPhysReg, 16> Hints = {1, 2, 3};
  SmallVector<MCPhysReg, 16> Order = {4, 1, 5, 6};
  AllocationOrder O(std::move(Hints), Order, false);
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4, 5, 6}), loadOrder(O));
}

TEST(AllocationOrderTest, HardHints) {
  SmallVector<MCPhysReg, 16> Hints = {1, 2, 3};
  SmallVector<MCPhysReg, 16> Order = {4, 5, 6, 7};
  AllocationOrder O(std::move(Hints), Order, true);
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3}), loadOrder(O));
}

TEST(AllocationOrderTest, LimitsBasic) {
  SmallVector<MCPhysReg, 16> Hints = {1, 2, 3};
  SmallVector<MCPhysReg, 16> Order = {4, 5, 6, 7};
  AllocationOrder O(std::move(Hints), Order, false);
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4, 5, 6, 7}), loadOrder(O, 0));
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4}), loadOrder(O, 1));
}

TEST(AllocationOrderTest, LimitsDuplicates) {
  SmallVector<MCPhysReg, 16> Hints = {1, 2, 3};
  SmallVector<MCPhysReg, 16> Order = {4, 1, 5, 6};
  AllocationOrder O(std::move(Hints), Order, false);
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4}), loadOrder(O, 1));
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4}), loadOrder(O, 2));
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4, 5}), loadOrder(O, 3));
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4, 5, 6}), loadOrder(O, 4));
}

TEST(AllocationOrderTest, LimitsHardHints) {
  SmallVector<MCPhysReg, 16> Hints = {1, 2, 3};
  SmallVector<MCPhysReg, 16> Order = {4, 1, 5, 6};
  AllocationOrder O(std::move(Hints), Order, true);
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3}), loadOrder(O, 1));
}

TEST(AllocationOrderTest, DuplicateIsFirst) {
  SmallVector<MCPhysReg, 16> Hints = {1, 2, 3};
  SmallVector<MCPhysReg, 16> Order = {1, 4, 5, 6};
  AllocationOrder O(std::move(Hints), Order, false);
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4, 5, 6}), loadOrder(O));
}

TEST(AllocationOrderTest, DuplicateIsFirstWithLimits) {
  SmallVector<MCPhysReg, 16> Hints = {1, 2, 3};
  SmallVector<MCPhysReg, 16> Order = {1, 4, 5, 6};
  AllocationOrder O(std::move(Hints), Order, false);
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3}), loadOrder(O, 1));
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4}), loadOrder(O, 2));
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4, 5}), loadOrder(O, 3));
}

TEST(AllocationOrderTest, NoHints) {
  SmallVector<MCPhysReg, 16> Hints;
  SmallVector<MCPhysReg, 16> Order = {1, 2, 3, 4};
  AllocationOrder O(std::move(Hints), Order, false);
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3, 4}), loadOrder(O));
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2}), loadOrder(O, 2));
  EXPECT_EQ((std::vector<MCPhysReg>{1, 2, 3}), loadOrder(O, 3));
}

TEST(AllocationOrderTest, IsHintTest) {
  SmallVector<MCPhysReg, 16> Hints = {1, 2, 3};
  SmallVector<MCPhysReg, 16> Order = {4, 1, 5, 6};
  AllocationOrder O(std::move(Hints), Order, false);
  O.rewind();
  auto V = O.next();
  EXPECT_TRUE(O.isHint());
  EXPECT_EQ(V, 1U);
  O.next();
  EXPECT_TRUE(O.isHint());
  O.next();
  EXPECT_TRUE(O.isHint());
  V = O.next();
  EXPECT_FALSE(O.isHint());
  EXPECT_EQ(V, 4U);
  V = O.next();
  EXPECT_TRUE(O.isHint(1));
  EXPECT_FALSE(O.isHint());
  EXPECT_EQ(V, 5U);
}
