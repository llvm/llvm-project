//===-- StringLexerTest.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StringLexer.h"
#include "gtest/gtest.h"

using namespace lldb_utility;

TEST(StringLexerTest, GetUnlexed) {
  StringLexer l("foo");
  EXPECT_EQ("foo", l.GetUnlexed());
  l.Next();
  EXPECT_EQ("oo", l.GetUnlexed());
  l.Next();
  l.Next();
  EXPECT_EQ("", l.GetUnlexed());
}

TEST(StringLexerTest, HasAtLeast) {
  StringLexer l("foo");
  EXPECT_FALSE(l.HasAtLeast(5));
  EXPECT_FALSE(l.HasAtLeast(4));
  EXPECT_TRUE(l.HasAtLeast(3));
  EXPECT_TRUE(l.HasAtLeast(2));
  EXPECT_TRUE(l.HasAtLeast(1));

  l.Next();
  EXPECT_FALSE(l.HasAtLeast(5));
  EXPECT_FALSE(l.HasAtLeast(4));
  EXPECT_FALSE(l.HasAtLeast(3));
  EXPECT_TRUE(l.HasAtLeast(2));
  EXPECT_TRUE(l.HasAtLeast(1));

  l.Next();
  l.Next();
  EXPECT_FALSE(l.HasAtLeast(5));
  EXPECT_FALSE(l.HasAtLeast(4));
  EXPECT_FALSE(l.HasAtLeast(3));
  EXPECT_FALSE(l.HasAtLeast(2));
  EXPECT_FALSE(l.HasAtLeast(1));
}

TEST(StringLexerTest, AdvanceIf) {
  StringLexer l("foobar");

  EXPECT_FALSE(l.AdvanceIf("oo"));
  // Skip the "fo" part.
  EXPECT_TRUE(l.AdvanceIf("fo"));
  EXPECT_FALSE(l.AdvanceIf("obarz"));
  // Skip the remaining string.
  EXPECT_TRUE(l.AdvanceIf("obar"));

  EXPECT_FALSE(l.AdvanceIf("obarz"));
  EXPECT_FALSE(l.AdvanceIf("foo"));
  EXPECT_FALSE(l.AdvanceIf("o"));
  EXPECT_FALSE(l.AdvanceIf(" "));
}

TEST(StringLexerTest, PutBack) {
  StringLexer l("foo");

  l.Next();
  l.PutBack(1);
  EXPECT_EQ("foo", l.GetUnlexed());

  l.Next();
  l.Next();
  l.Next();
  l.PutBack(2);
  EXPECT_EQ("oo", l.GetUnlexed());

  l.PutBack(1);
  EXPECT_EQ("foo", l.GetUnlexed());
}

TEST(StringLexerTest, Peek) {
  StringLexer l("foo");

  EXPECT_EQ('f', l.Peek());
  l.Next();
  EXPECT_EQ('o', l.Peek());
  l.Next();
  EXPECT_EQ('o', l.Peek());
}

TEST(StringLexerTest, Next) {
  StringLexer l("foo");
  EXPECT_EQ('f', l.Next());
  EXPECT_EQ('o', l.Next());
  EXPECT_EQ('o', l.Next());
}

TEST(StringLexerTest, NextIf) {
  StringLexer l("foo");

  EXPECT_FALSE(l.NextIf('\0'));
  EXPECT_FALSE(l.NextIf(' '));
  EXPECT_FALSE(l.NextIf('o'));

  EXPECT_TRUE(l.NextIf('f'));

  EXPECT_FALSE(l.NextIf('\0'));
  EXPECT_FALSE(l.NextIf(' '));
  EXPECT_FALSE(l.NextIf('f'));

  EXPECT_TRUE(l.NextIf('o'));

  EXPECT_FALSE(l.NextIf('\0'));
  EXPECT_FALSE(l.NextIf(' '));
  EXPECT_FALSE(l.NextIf('f'));

  EXPECT_TRUE(l.NextIf('o'));
}

TEST(StringLexerTest, NextIfList) {
  StringLexer l("foo");

  EXPECT_FALSE(l.NextIf({'\0', ' ', 'o'}).first);

  auto r = l.NextIf({'f'});
  EXPECT_TRUE(r.first);
  EXPECT_EQ('f', r.second);

  EXPECT_FALSE(l.NextIf({'\0', ' ', 'f'}).first);

  r = l.NextIf({'f', 'o'});
  EXPECT_TRUE(r.first);
  EXPECT_EQ('o', r.second);

  EXPECT_FALSE(l.NextIf({'\0', ' ', 'f'}).first);

  r = l.NextIf({'*', 'f', 'o', 'o'});
  EXPECT_TRUE(r.first);
  EXPECT_EQ('o', r.second);
}
