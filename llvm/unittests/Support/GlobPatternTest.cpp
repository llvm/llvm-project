//===- llvm/unittest/Support/GlobPatternTest.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/GlobPattern.h"
#include "gtest/gtest.h"

using namespace llvm;
namespace {

class GlobPatternTest : public ::testing::Test {};

TEST_F(GlobPatternTest, Empty) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match(""));
  EXPECT_FALSE(Pat1->match("a"));
}

TEST_F(GlobPatternTest, Glob) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("ab*c*def");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("abcdef"));
  EXPECT_TRUE(Pat1->match("abxcxdef"));
  EXPECT_FALSE(Pat1->match(""));
  EXPECT_FALSE(Pat1->match("xabcdef"));
  EXPECT_FALSE(Pat1->match("abcdefx"));
}

TEST_F(GlobPatternTest, Wildcard) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("a??c");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("axxc"));
  EXPECT_FALSE(Pat1->match("axxx"));
  EXPECT_FALSE(Pat1->match(""));
}

TEST_F(GlobPatternTest, Escape) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("\\*");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("*"));
  EXPECT_FALSE(Pat1->match("\\*"));
  EXPECT_FALSE(Pat1->match("a"));

  Expected<GlobPattern> Pat2 = GlobPattern::create("a?\\?c");
  EXPECT_TRUE((bool)Pat2);
  EXPECT_TRUE(Pat2->match("ax?c"));
  EXPECT_FALSE(Pat2->match("axxc"));
  EXPECT_FALSE(Pat2->match(""));
}

TEST_F(GlobPatternTest, BasicCharacterClass) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("[abc-fy-z]");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("a"));
  EXPECT_TRUE(Pat1->match("b"));
  EXPECT_TRUE(Pat1->match("c"));
  EXPECT_TRUE(Pat1->match("d"));
  EXPECT_TRUE(Pat1->match("e"));
  EXPECT_TRUE(Pat1->match("f"));
  EXPECT_TRUE(Pat1->match("y"));
  EXPECT_TRUE(Pat1->match("z"));
  EXPECT_FALSE(Pat1->match("g"));
  EXPECT_FALSE(Pat1->match(""));
}

TEST_F(GlobPatternTest, NegatedCharacterClass) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("[^abc-fy-z]");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("g"));
  EXPECT_FALSE(Pat1->match("a"));
  EXPECT_FALSE(Pat1->match("b"));
  EXPECT_FALSE(Pat1->match("c"));
  EXPECT_FALSE(Pat1->match("d"));
  EXPECT_FALSE(Pat1->match("e"));
  EXPECT_FALSE(Pat1->match("f"));
  EXPECT_FALSE(Pat1->match("y"));
  EXPECT_FALSE(Pat1->match("z"));
  EXPECT_FALSE(Pat1->match(""));

  Expected<GlobPattern> Pat2 = GlobPattern::create("[!abc-fy-z]");
  EXPECT_TRUE((bool)Pat2);
  EXPECT_TRUE(Pat2->match("g"));
  EXPECT_FALSE(Pat2->match("a"));
  EXPECT_FALSE(Pat2->match("b"));
  EXPECT_FALSE(Pat2->match("c"));
  EXPECT_FALSE(Pat2->match("d"));
  EXPECT_FALSE(Pat2->match("e"));
  EXPECT_FALSE(Pat2->match("f"));
  EXPECT_FALSE(Pat2->match("y"));
  EXPECT_FALSE(Pat2->match("z"));
  EXPECT_FALSE(Pat2->match(""));
}

TEST_F(GlobPatternTest, BracketFrontOfCharacterClass) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("[]a]x");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("]x"));
  EXPECT_TRUE(Pat1->match("ax"));
  EXPECT_FALSE(Pat1->match("a]x"));
  EXPECT_FALSE(Pat1->match(""));
}

TEST_F(GlobPatternTest, SpecialCharsInCharacterClass) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("[*?^]");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("*"));
  EXPECT_TRUE(Pat1->match("?"));
  EXPECT_TRUE(Pat1->match("^"));
  EXPECT_FALSE(Pat1->match("*?^"));
  EXPECT_FALSE(Pat1->match(""));
}

TEST_F(GlobPatternTest, Invalid) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("[");
  EXPECT_FALSE((bool)Pat1);
  handleAllErrors(Pat1.takeError(), [&](ErrorInfoBase &EIB) {});

  Expected<GlobPattern> Pat2 = GlobPattern::create("[]");
  EXPECT_FALSE((bool)Pat2);
  handleAllErrors(Pat2.takeError(), [&](ErrorInfoBase &EIB) {});
}

TEST_F(GlobPatternTest, ExtSym) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("a*\xFF");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("axxx\xFF"));
  Expected<GlobPattern> Pat2 = GlobPattern::create("[\xFF-\xFF]");
  EXPECT_TRUE((bool)Pat2);
  EXPECT_TRUE(Pat2->match("\xFF"));
}

TEST_F(GlobPatternTest, IsTrivialMatchAll) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("*");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->isTrivialMatchAll());

  const char *NegativeCases[] = {"a*", "*a", "?*", "*?", "**", "\\*"};
  for (auto *P : NegativeCases) {
    Expected<GlobPattern> Pat2 = GlobPattern::create(P);
    EXPECT_TRUE((bool)Pat2);
    EXPECT_FALSE(Pat2->isTrivialMatchAll());
  }
}
}
