//===- SequenceTest.cpp - Unit tests for a sequence abstraciton -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Sequence.h"
#include "gtest/gtest.h"

#include <list>

using namespace llvm;

namespace {

TEST(SequenceTest, Forward) {
  int X = 0;
  for (int I : seq(0, 10)) {
    EXPECT_EQ(X, I);
    ++X;
  }
  EXPECT_EQ(10, X);
}

TEST(SequenceTest, Backward) {
  int X = 9;
  for (int I : reverse(seq(0, 10))) {
    EXPECT_EQ(X, I);
    --X;
  }
  EXPECT_EQ(-1, X);
}

TEST(SequenceTest, Distance) {
  const auto Forward = seq(0, 10);
  EXPECT_EQ(std::distance(Forward.begin(), Forward.end()), 10);
  EXPECT_EQ(std::distance(Forward.rbegin(), Forward.rend()), 10);
}

TEST(SequenceTest, Dereference) {
  const auto Forward = seq(0, 10).begin();
  EXPECT_EQ(Forward[0], 0);
  EXPECT_EQ(Forward[2], 2);
  const auto Backward = seq(0, 10).rbegin();
  EXPECT_EQ(Backward[0], 9);
  EXPECT_EQ(Backward[2], 7);
}

} // anonymous namespace
