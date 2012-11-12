//===- llvm/unittest/Support/RegexTest.cpp - Regex tests --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Support/Regex.h"
#include "llvm/ADT/SmallVector.h"
#include <cstring>

using namespace llvm;
namespace {

class RegexTest : public ::testing::Test {
};

TEST_F(RegexTest, Basics) {
  Regex r1("^[0-9]+$");
  EXPECT_TRUE(r1.match("916"));
  EXPECT_TRUE(r1.match("9"));
  EXPECT_FALSE(r1.match("9a"));

  SmallVector<StringRef, 1> Matches;
  Regex r2("[0-9]+");
  EXPECT_TRUE(r2.match("aa216b", &Matches));
  EXPECT_EQ(1u, Matches.size());
  EXPECT_EQ("216", Matches[0].str());

  Regex r3("[0-9]+([a-f])?:([0-9]+)");
  EXPECT_TRUE(r3.match("9a:513b", &Matches));
  EXPECT_EQ(3u, Matches.size());
  EXPECT_EQ("9a:513", Matches[0].str());
  EXPECT_EQ("a", Matches[1].str());
  EXPECT_EQ("513", Matches[2].str());

  EXPECT_TRUE(r3.match("9:513b", &Matches));
  EXPECT_EQ(3u, Matches.size());
  EXPECT_EQ("9:513", Matches[0].str());
  EXPECT_EQ("", Matches[1].str());
  EXPECT_EQ("513", Matches[2].str());

  Regex r4("a[^b]+b");
  std::string String="axxb";
  String[2] = '\0';
  EXPECT_FALSE(r4.match("abb"));
  EXPECT_TRUE(r4.match(String, &Matches));
  EXPECT_EQ(1u, Matches.size());
  EXPECT_EQ(String, Matches[0].str());


  std::string NulPattern="X[0-9]+X([a-f])?:([0-9]+)";
  String="YX99a:513b";
  NulPattern[7] = '\0';
  Regex r5(NulPattern);
  EXPECT_FALSE(r5.match(String));
  EXPECT_FALSE(r5.match("X9"));
  String[3]='\0';
  EXPECT_TRUE(r5.match(String));
}

TEST_F(RegexTest, Substitution) {
  std::string Error;

  EXPECT_EQ("aNUMber", Regex("[0-9]+").sub("NUM", "a1234ber"));

  // Standard Escapes
  EXPECT_EQ("a\\ber", Regex("[0-9]+").sub("\\\\", "a1234ber", &Error));
  EXPECT_EQ(Error, "");
  EXPECT_EQ("a\nber", Regex("[0-9]+").sub("\\n", "a1234ber", &Error));
  EXPECT_EQ(Error, "");
  EXPECT_EQ("a\tber", Regex("[0-9]+").sub("\\t", "a1234ber", &Error));
  EXPECT_EQ(Error, "");
  EXPECT_EQ("ajber", Regex("[0-9]+").sub("\\j", "a1234ber", &Error));
  EXPECT_EQ(Error, "");

  EXPECT_EQ("aber", Regex("[0-9]+").sub("\\", "a1234ber", &Error));
  EXPECT_EQ(Error, "replacement string contained trailing backslash");
  
  // Backreferences
  EXPECT_EQ("aa1234bber", Regex("a[0-9]+b").sub("a\\0b", "a1234ber", &Error));
  EXPECT_EQ(Error, "");

  EXPECT_EQ("a1234ber", Regex("a([0-9]+)b").sub("a\\1b", "a1234ber", &Error));
  EXPECT_EQ(Error, "");

  EXPECT_EQ("aber", Regex("a[0-9]+b").sub("a\\100b", "a1234ber", &Error));
  EXPECT_EQ(Error, "invalid backreference string '100'");
}

}
