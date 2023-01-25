//===- llvm/unittest/DebugInfo/LogicalView/StringPoolTest.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVStringPool.h"
#include <vector>

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::logicalview;

namespace {

TEST(StringPoolTest, AddStrings) {
  LVStringPool PoolInstance;
  EXPECT_EQ(0u, PoolInstance.getSize());

  // Get indexes for the initial strings.
  EXPECT_EQ(1u, PoolInstance.getIndex("one"));
  EXPECT_EQ(2u, PoolInstance.getIndex("two"));
  EXPECT_EQ(3u, PoolInstance.getIndex("three"));
  EXPECT_EQ(4u, PoolInstance.getIndex("four"));
  EXPECT_EQ(5u, PoolInstance.getIndex("five"));
  EXPECT_EQ(5u, PoolInstance.getSize());

  // Verify the string returned by the given index.
  EXPECT_EQ("one", PoolInstance.getString(1));
  EXPECT_EQ("two", PoolInstance.getString(2));
  EXPECT_EQ("three", PoolInstance.getString(3));
  EXPECT_EQ("four", PoolInstance.getString(4));
  EXPECT_EQ("five", PoolInstance.getString(5));
  EXPECT_EQ(5u, PoolInstance.getSize());

  // Get indexes for the same initial strings.
  EXPECT_EQ(5u, PoolInstance.getIndex("five"));
  EXPECT_EQ(4u, PoolInstance.getIndex("four"));
  EXPECT_EQ(3u, PoolInstance.getIndex("three"));
  EXPECT_EQ(2u, PoolInstance.getIndex("two"));
  EXPECT_EQ(1u, PoolInstance.getIndex("one"));
  EXPECT_EQ(5u, PoolInstance.getSize());

  // Empty string gets the index zero.
  EXPECT_EQ(0u, PoolInstance.getIndex(""));
  EXPECT_EQ(5u, PoolInstance.getSize());

  // Empty string for invalid index.
  EXPECT_EQ("", PoolInstance.getString(620));

  // Lookup for strings
  EXPECT_EQ(5u, PoolInstance.findIndex("five"));
  EXPECT_TRUE(PoolInstance.isValidIndex(PoolInstance.findIndex("five")));
  EXPECT_FALSE(PoolInstance.isValidIndex(PoolInstance.findIndex("FIVE")));
}

} // namespace
