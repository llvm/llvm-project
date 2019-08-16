//===- TrigramIndexTest.cpp - Unit tests for TrigramIndex -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TrigramIndex.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace llvm;

namespace {

class TrigramIndexTest : public ::testing::Test {
protected:
  std::unique_ptr<TrigramIndex> makeTrigramIndex(
      std::vector<std::string> Rules) {
    std::unique_ptr<TrigramIndex> TI =
        make_unique<TrigramIndex>();
    for (auto &Rule : Rules)
      TI->insert(Rule);
    return TI;
  }
};

TEST_F(TrigramIndexTest, Empty) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({});
  EXPECT_FALSE(TI->isDefeated());
  EXPECT_TRUE(TI->isDefinitelyOut("foo"));
}

TEST_F(TrigramIndexTest, Basic) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"*hello*", "*wor.d*"});
  EXPECT_FALSE(TI->isDefeated());
  EXPECT_TRUE(TI->isDefinitelyOut("foo"));
}

TEST_F(TrigramIndexTest, NoTrigramsInRules) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"b.r", "za*az"});
  EXPECT_TRUE(TI->isDefeated());
  EXPECT_FALSE(TI->isDefinitelyOut("foo"));
  EXPECT_FALSE(TI->isDefinitelyOut("bar"));
  EXPECT_FALSE(TI->isDefinitelyOut("zakaz"));
}

TEST_F(TrigramIndexTest, NoTrigramsInARule) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"*hello*", "*wo.ld*"});
  EXPECT_TRUE(TI->isDefeated());
  EXPECT_FALSE(TI->isDefinitelyOut("foo"));
}

TEST_F(TrigramIndexTest, RepetitiveRule) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"*bar*bar*bar*bar*bar", "bar*bar"});
  EXPECT_FALSE(TI->isDefeated());
  EXPECT_TRUE(TI->isDefinitelyOut("foo"));
  EXPECT_TRUE(TI->isDefinitelyOut("bar"));
  EXPECT_FALSE(TI->isDefinitelyOut("barbara"));
  EXPECT_FALSE(TI->isDefinitelyOut("bar+bar"));
}

TEST_F(TrigramIndexTest, PopularTrigram) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"*aaa*", "*aaaa*", "*aaaaa*", "*aaaaa*", "*aaaaaa*"});
  EXPECT_TRUE(TI->isDefeated());
}

TEST_F(TrigramIndexTest, PopularTrigram2) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"class1.h", "class2.h", "class3.h", "class4.h", "class.h"});
  EXPECT_TRUE(TI->isDefeated());
}

TEST_F(TrigramIndexTest, TooComplicatedRegex) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"[0-9]+"});
  EXPECT_TRUE(TI->isDefeated());
}

TEST_F(TrigramIndexTest, TooComplicatedRegex2) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"foo|bar"});
  EXPECT_TRUE(TI->isDefeated());
}

TEST_F(TrigramIndexTest, EscapedSymbols) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"*c\\+\\+*", "*hello\\\\world*", "a\\tb", "a\\0b"});
  EXPECT_FALSE(TI->isDefeated());
  EXPECT_FALSE(TI->isDefinitelyOut("c++"));
  EXPECT_TRUE(TI->isDefinitelyOut("c\\+\\+"));
  EXPECT_FALSE(TI->isDefinitelyOut("hello\\world"));
  EXPECT_TRUE(TI->isDefinitelyOut("hello\\\\world"));
  EXPECT_FALSE(TI->isDefinitelyOut("atb"));
  EXPECT_TRUE(TI->isDefinitelyOut("a\\tb"));
  EXPECT_TRUE(TI->isDefinitelyOut("a\tb"));
  EXPECT_FALSE(TI->isDefinitelyOut("a0b"));
}

TEST_F(TrigramIndexTest, Backreference1) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"*foo\\1*"});
  EXPECT_TRUE(TI->isDefeated());
}

TEST_F(TrigramIndexTest, Backreference2) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"*foo\\2*"});
  EXPECT_TRUE(TI->isDefeated());
}

TEST_F(TrigramIndexTest, Sequence) {
  std::unique_ptr<TrigramIndex> TI =
      makeTrigramIndex({"class1.h", "class2.h", "class3.h", "class4.h"});
  EXPECT_FALSE(TI->isDefeated());
  EXPECT_FALSE(TI->isDefinitelyOut("class1"));
  EXPECT_TRUE(TI->isDefinitelyOut("class.h"));
  EXPECT_TRUE(TI->isDefinitelyOut("class"));
}

}  // namespace
