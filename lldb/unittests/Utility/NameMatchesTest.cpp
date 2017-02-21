//===-- NameMatchesTest.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NameMatches.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(NameMatchesTest, Ignore) {
  EXPECT_TRUE(NameMatches("foo", NameMatch::Ignore, "bar"));
}

TEST(NameMatchesTest, Equals) {
  EXPECT_TRUE(NameMatches("foo", NameMatch::Equals, "foo"));
  EXPECT_FALSE(NameMatches("foo", NameMatch::Equals, "bar"));
}

TEST(NameMatchesTest, Contains) {
  EXPECT_TRUE(NameMatches("foobar", NameMatch::Contains, "foo"));
  EXPECT_TRUE(NameMatches("foobar", NameMatch::Contains, "oob"));
  EXPECT_TRUE(NameMatches("foobar", NameMatch::Contains, "bar"));
  EXPECT_TRUE(NameMatches("foobar", NameMatch::Contains, "foobar"));
  EXPECT_TRUE(NameMatches("", NameMatch::Contains, ""));
  EXPECT_FALSE(NameMatches("", NameMatch::Contains, "foo"));
  EXPECT_FALSE(NameMatches("foobar", NameMatch::Contains, "baz"));
}

TEST(NameMatchesTest, StartsWith) {
  EXPECT_TRUE(NameMatches("foo", NameMatch::StartsWith, "f"));
  EXPECT_TRUE(NameMatches("foo", NameMatch::StartsWith, ""));
  EXPECT_TRUE(NameMatches("", NameMatch::StartsWith, ""));
  EXPECT_FALSE(NameMatches("foo", NameMatch::StartsWith, "b"));
  EXPECT_FALSE(NameMatches("", NameMatch::StartsWith, "b"));
}

TEST(NameMatchesTest, EndsWith) {
  EXPECT_TRUE(NameMatches("foo", NameMatch::EndsWith, "o"));
  EXPECT_TRUE(NameMatches("foo", NameMatch::EndsWith, ""));
  EXPECT_TRUE(NameMatches("", NameMatch::EndsWith, ""));
  EXPECT_FALSE(NameMatches("foo", NameMatch::EndsWith, "b"));
  EXPECT_FALSE(NameMatches("", NameMatch::EndsWith, "b"));
}

TEST(NameMatchesTest, RegularExpression) {
  EXPECT_TRUE(NameMatches("foobar", NameMatch::RegularExpression, "foo"));
  EXPECT_TRUE(NameMatches("foobar", NameMatch::RegularExpression, "f[oa]o"));
  EXPECT_TRUE(NameMatches("foo", NameMatch::RegularExpression, ""));
  EXPECT_TRUE(NameMatches("", NameMatch::RegularExpression, ""));
  EXPECT_FALSE(NameMatches("foo", NameMatch::RegularExpression, "b"));
  EXPECT_FALSE(NameMatches("", NameMatch::RegularExpression, "b"));
  EXPECT_FALSE(NameMatches("^a", NameMatch::RegularExpression, "^a"));
}
