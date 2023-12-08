//===- VersionTupleTests.cpp - Version Number Handling Tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/VersionTuple.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(VersionTuple, getAsString) {
  EXPECT_EQ("0", VersionTuple().getAsString());
  EXPECT_EQ("1", VersionTuple(1).getAsString());
  EXPECT_EQ("1.2", VersionTuple(1, 2).getAsString());
  EXPECT_EQ("1.2.3", VersionTuple(1, 2, 3).getAsString());
  EXPECT_EQ("1.2.3.4", VersionTuple(1, 2, 3, 4).getAsString());
}

TEST(VersionTuple, tryParse) {
  VersionTuple VT;

  EXPECT_FALSE(VT.tryParse("1"));
  EXPECT_EQ("1", VT.getAsString());

  EXPECT_FALSE(VT.tryParse("1.2"));
  EXPECT_EQ("1.2", VT.getAsString());

  EXPECT_FALSE(VT.tryParse("1.2.3"));
  EXPECT_EQ("1.2.3", VT.getAsString());

  EXPECT_FALSE(VT.tryParse("1.2.3.4"));
  EXPECT_EQ("1.2.3.4", VT.getAsString());

  EXPECT_TRUE(VT.tryParse(""));
  EXPECT_TRUE(VT.tryParse("1."));
  EXPECT_TRUE(VT.tryParse("1.2."));
  EXPECT_TRUE(VT.tryParse("1.2.3."));
  EXPECT_TRUE(VT.tryParse("1.2.3.4."));
  EXPECT_TRUE(VT.tryParse("1.2.3.4.5"));
  EXPECT_TRUE(VT.tryParse("1-2"));
  EXPECT_TRUE(VT.tryParse("1+2"));
  EXPECT_TRUE(VT.tryParse(".1"));
  EXPECT_TRUE(VT.tryParse(" 1"));
  EXPECT_TRUE(VT.tryParse("1 "));
  EXPECT_TRUE(VT.tryParse("."));
}
