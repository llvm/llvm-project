//===-- RegularExpressionTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RegularExpression.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;

TEST(RegularExpression, Valid) {
  RegularExpression r1("^[0-9]+$");
  cantFail(r1.GetError());
  EXPECT_TRUE(r1.IsValid());
  EXPECT_EQ("^[0-9]+$", r1.GetText());
  EXPECT_TRUE(r1.Execute("916"));
}

TEST(RegularExpression, CopyAssignment) {
  RegularExpression r1("^[0-9]+$");
  RegularExpression r2 = r1;
  cantFail(r2.GetError());
  EXPECT_TRUE(r2.IsValid());
  EXPECT_EQ("^[0-9]+$", r2.GetText());
  EXPECT_TRUE(r2.Execute("916"));
}

TEST(RegularExpression, Empty) {
  RegularExpression r1("");
  Error err = r1.GetError();
  EXPECT_TRUE(static_cast<bool>(err));
  consumeError(std::move(err));
  EXPECT_FALSE(r1.IsValid());
  EXPECT_EQ("", r1.GetText());
  EXPECT_FALSE(r1.Execute("916"));
}

TEST(RegularExpression, Invalid) {
  RegularExpression r1("a[b-");
  Error err = r1.GetError();
  EXPECT_TRUE(static_cast<bool>(err));
  consumeError(std::move(err));
  EXPECT_FALSE(r1.IsValid());
  EXPECT_EQ("a[b-", r1.GetText());
  EXPECT_FALSE(r1.Execute("ab"));
}

TEST(RegularExpression, Match) {
  RegularExpression r1("[0-9]+([a-f])?:([0-9]+)");
  cantFail(r1.GetError());
  EXPECT_TRUE(r1.IsValid());
  EXPECT_EQ("[0-9]+([a-f])?:([0-9]+)", r1.GetText());

  SmallVector<StringRef, 3> matches;
  EXPECT_TRUE(r1.Execute("9a:513b", &matches));
  EXPECT_EQ(3u, matches.size());
  EXPECT_EQ("9a:513", matches[0].str());
  EXPECT_EQ("a", matches[1].str());
  EXPECT_EQ("513", matches[2].str());
}
