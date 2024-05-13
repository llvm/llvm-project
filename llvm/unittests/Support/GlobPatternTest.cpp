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

  auto Pat3 = GlobPattern::create("\\{");
  ASSERT_TRUE((bool)Pat3);
  EXPECT_TRUE(Pat3->match("{"));
  EXPECT_FALSE(Pat3->match("\\{"));
  EXPECT_FALSE(Pat3->match(""));

  auto Pat4 = GlobPattern::create("\\a");
  ASSERT_TRUE((bool)Pat4);
  EXPECT_TRUE(Pat4->match("a"));
  EXPECT_FALSE(Pat4->match("\\a"));

  for (size_t I = 0; I != 4; ++I) {
    std::string S(I, '\\');
    Expected<GlobPattern> Pat = GlobPattern::create(S);
    if (I % 2) {
      EXPECT_FALSE((bool)Pat);
      handleAllErrors(Pat.takeError(), [&](ErrorInfoBase &) {});
    } else {
      EXPECT_TRUE((bool)Pat);
    }
  }
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

  Expected<GlobPattern> Pat2 = GlobPattern::create("[ab]*[cd]?**[ef]");
  ASSERT_TRUE((bool)Pat2);
  EXPECT_TRUE(Pat2->match("aecde"));
  EXPECT_FALSE(Pat2->match("aecdg"));
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
  auto Pat1 = GlobPattern::create("[*?^{},]");
  ASSERT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("*"));
  EXPECT_TRUE(Pat1->match("?"));
  EXPECT_TRUE(Pat1->match("^"));
  EXPECT_TRUE(Pat1->match("{"));
  EXPECT_TRUE(Pat1->match("}"));
  EXPECT_TRUE(Pat1->match(","));
  EXPECT_FALSE(Pat1->match("*?^{},"));
  EXPECT_FALSE(Pat1->match(""));

  Expected<GlobPattern> Pat2 = GlobPattern::create("[*]");
  ASSERT_TRUE((bool)Pat2);
  EXPECT_TRUE(Pat2->match("*"));
  EXPECT_FALSE(Pat2->match("]"));
}

TEST_F(GlobPatternTest, Invalid) {
  for (const auto &InvalidPattern : {"[", "[]"}) {
    auto Pat1 = GlobPattern::create(InvalidPattern);
    EXPECT_FALSE((bool)Pat1) << "Expected invalid pattern: " << InvalidPattern;
    handleAllErrors(Pat1.takeError(), [&](ErrorInfoBase &EIB) {});
  }
}

TEST_F(GlobPatternTest, InvalidBraceExpansion) {
  for (const auto &InvalidPattern :
       {"{", "{{", "{\\", "{\\}", "{}", "{a}", "[{}"}) {
    auto Pat1 = GlobPattern::create(InvalidPattern, /*MaxSubPatterns=*/1024);
    EXPECT_FALSE((bool)Pat1) << "Expected invalid pattern: " << InvalidPattern;
    handleAllErrors(Pat1.takeError(), [&](ErrorInfoBase &EIB) {});
  }
  auto Pat1 = GlobPattern::create("{a,b}{c,d}{e,f}", /*MaxSubPatterns=*/7);
  EXPECT_FALSE((bool)Pat1);
  handleAllErrors(Pat1.takeError(), [&](ErrorInfoBase &EIB) {});
}

TEST_F(GlobPatternTest, BraceExpansion) {
  auto Pat1 = GlobPattern::create("{a,b}{1,2}", /*MaxSubPatterns=*/1024);
  ASSERT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("a1"));
  EXPECT_TRUE(Pat1->match("a2"));
  EXPECT_TRUE(Pat1->match("b1"));
  EXPECT_TRUE(Pat1->match("b2"));
  EXPECT_FALSE(Pat1->match("ab"));

  auto Pat2 = GlobPattern::create(",}{foo,\\,\\},z*}", /*MaxSubPatterns=*/1024);
  ASSERT_TRUE((bool)Pat2);
  EXPECT_TRUE(Pat2->match(",}foo"));
  EXPECT_TRUE(Pat2->match(",},}"));
  EXPECT_TRUE(Pat2->match(",}z"));
  EXPECT_TRUE(Pat2->match(",}zoo"));
  EXPECT_FALSE(Pat2->match(",}fooz"));
  EXPECT_FALSE(Pat2->match("foo"));
  EXPECT_FALSE(Pat2->match(""));

  // This test breaks if we store terms separately and attempt to match them one
  // by one instead of using subglobs
  auto Pat3 = GlobPattern::create("{a,ab}b", /*MaxSubPatterns=*/1024);
  ASSERT_TRUE((bool)Pat3);
  EXPECT_TRUE(Pat3->match("ab"));
  EXPECT_TRUE(Pat3->match("abb"));
}

TEST_F(GlobPatternTest, NoBraceExpansion) {
  auto Pat1 = GlobPattern::create("{a,b}{1,2}");
  ASSERT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match("{a,b}{1,2}"));
  EXPECT_FALSE(Pat1->match("a1"));

  auto Pat2 = GlobPattern::create("{{");
  ASSERT_TRUE((bool)Pat2);
  EXPECT_TRUE(Pat2->match("{{"));
}

TEST_F(GlobPatternTest, BraceExpansionCharacterClass) {
  // Matches mangled names of C++ standard library functions
  auto Pat =
      GlobPattern::create("_Z{N,NK,}S[tabsiod]*", /*MaxSubPatterns=*/1024);
  ASSERT_TRUE((bool)Pat);
  EXPECT_TRUE(Pat->match("_ZNSt6vectorIiSaIiEE9push_backEOi"));
  EXPECT_TRUE(Pat->match("_ZNKStfoo"));
  EXPECT_TRUE(Pat->match("_ZNSafoo"));
  EXPECT_TRUE(Pat->match("_ZStfoo"));
  EXPECT_FALSE(Pat->match("_Zfoo"));
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

TEST_F(GlobPatternTest, NUL) {
  for (char C : "?*") {
    std::string S(1, C);
    Expected<GlobPattern> Pat = GlobPattern::create(S);
    ASSERT_TRUE((bool)Pat);
    EXPECT_TRUE(Pat->match(S));
    if (C == '*') {
      EXPECT_TRUE(Pat->match(S + '\0'));
    } else {
      EXPECT_FALSE(Pat->match(S + '\0'));
      handleAllErrors(Pat.takeError(), [&](ErrorInfoBase &) {});
    }
  }
}

TEST_F(GlobPatternTest, Pathological) {
  std::string P, S(40, 'a');
  StringRef Pieces[] = {"a*", "[ba]*", "{b*,a*}*"};
  for (int I = 0; I != 30; ++I)
    P += Pieces[I % 3];
  Expected<GlobPattern> Pat = GlobPattern::create(P, /*MaxSubPatterns=*/1024);
  ASSERT_TRUE((bool)Pat);
  EXPECT_TRUE(Pat->match(S));
  P += 'b';
  Pat = GlobPattern::create(P, /*MaxSubPatterns=*/1024);
  ASSERT_TRUE((bool)Pat);
  EXPECT_FALSE(Pat->match(S));
  EXPECT_TRUE(Pat->match(S + 'b'));
}
}
