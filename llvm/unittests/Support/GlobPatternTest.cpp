//===- llvm/unittest/Support/GlobPatternTest.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/GlobPattern.h"
#include "gtest/gtest.h"

using namespace llvm;
namespace {

class GlobPatternTest : public ::testing::Test {};

TEST_F(GlobPatternTest, Basics) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("");
  EXPECT_TRUE((bool)Pat1);
  EXPECT_TRUE(Pat1->match(""));
  EXPECT_FALSE(Pat1->match("a"));

  Expected<GlobPattern> Pat2 = GlobPattern::create("ab*c*def");
  EXPECT_TRUE((bool)Pat2);
  EXPECT_TRUE(Pat2->match("abcdef"));
  EXPECT_TRUE(Pat2->match("abxcxdef"));
  EXPECT_FALSE(Pat2->match(""));
  EXPECT_FALSE(Pat2->match("xabcdef"));
  EXPECT_FALSE(Pat2->match("abcdefx"));

  Expected<GlobPattern> Pat3 = GlobPattern::create("a??c");
  EXPECT_TRUE((bool)Pat3);
  EXPECT_TRUE(Pat3->match("axxc"));
  EXPECT_FALSE(Pat3->match("axxx"));
  EXPECT_FALSE(Pat3->match(""));

  Expected<GlobPattern> Pat4 = GlobPattern::create("[abc-fy-z]");
  EXPECT_TRUE((bool)Pat4);
  EXPECT_TRUE(Pat4->match("a"));
  EXPECT_TRUE(Pat4->match("b"));
  EXPECT_TRUE(Pat4->match("c"));
  EXPECT_TRUE(Pat4->match("d"));
  EXPECT_TRUE(Pat4->match("e"));
  EXPECT_TRUE(Pat4->match("f"));
  EXPECT_TRUE(Pat4->match("y"));
  EXPECT_TRUE(Pat4->match("z"));
  EXPECT_FALSE(Pat4->match("g"));
  EXPECT_FALSE(Pat4->match(""));

  Expected<GlobPattern> Pat5 = GlobPattern::create("[^abc-fy-z]");
  EXPECT_TRUE((bool)Pat5);
  EXPECT_TRUE(Pat5->match("g"));
  EXPECT_FALSE(Pat5->match("a"));
  EXPECT_FALSE(Pat5->match("b"));
  EXPECT_FALSE(Pat5->match("c"));
  EXPECT_FALSE(Pat5->match("d"));
  EXPECT_FALSE(Pat5->match("e"));
  EXPECT_FALSE(Pat5->match("f"));
  EXPECT_FALSE(Pat5->match("y"));
  EXPECT_FALSE(Pat5->match("z"));
  EXPECT_FALSE(Pat5->match(""));
}

TEST_F(GlobPatternTest, Invalid) {
  Expected<GlobPattern> Pat1 = GlobPattern::create("[");
  EXPECT_FALSE((bool)Pat1);
  handleAllErrors(Pat1.takeError(), [&](ErrorInfoBase &EIB) {});
}
}
