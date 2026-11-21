//===- SymbolLookupResultTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SymbolLookupResult APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/SymbolLookupResult.h"

#include "gtest/gtest.h"

using namespace orc_rt;

TEST(SymbolLookupResultTest, DefaultConstructedIsEmpty) {
  SymbolLookupResult R;
  EXPECT_TRUE(R.empty());
  EXPECT_EQ(R.size(), 0U);
  EXPECT_EQ(R.begin(), R.end());
}

TEST(SymbolLookupResultTest, PushBackIndexAndIterate) {
  int X = 0;
  SymbolLookupResult R;
  R.reserve(3);
  R.push_back(&X);
  R.push_back(nullptr);
  R.push_back(std::nullopt);
  ASSERT_EQ(R.size(), 3U);
  EXPECT_FALSE(R.empty());

  EXPECT_EQ(R[0], std::optional<void *>(&X));
  EXPECT_EQ(R[1], std::optional<void *>(nullptr));
  EXPECT_EQ(R[2], std::nullopt);

  R[2] = nullptr;
  size_t NumPresent = 0;
  for (auto &Addr : R)
    NumPresent += !!Addr;
  EXPECT_EQ(NumPresent, 3U);
}

TEST(SymbolLookupResultTest, ResizeFillsWithMissing) {
  SymbolLookupResult R;
  R.resize(2);
  ASSERT_EQ(R.size(), 2U);
  EXPECT_EQ(R[0], std::nullopt);
  EXPECT_EQ(R[1], std::nullopt);
}
