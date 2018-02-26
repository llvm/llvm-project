//===- llvm/unittest/Support/RegexTest.cpp - Regex tests --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Regex.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"
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

TEST_F(RegexTest, Backreferences) {
  Regex r1("([a-z]+)_\\1");
  SmallVector<StringRef, 4> Matches;
  EXPECT_TRUE(r1.match("abc_abc", &Matches));
  EXPECT_EQ(2u, Matches.size());
  EXPECT_FALSE(r1.match("abc_ab", &Matches));

  Regex r2("a([0-9])b\\1c\\1");
  EXPECT_TRUE(r2.match("a4b4c4", &Matches));
  EXPECT_EQ(2u, Matches.size());
  EXPECT_EQ("4", Matches[1].str());
  EXPECT_FALSE(r2.match("a2b2c3"));

  Regex r3("a([0-9])([a-z])b\\1\\2");
  EXPECT_TRUE(r3.match("a6zb6z", &Matches));
  EXPECT_EQ(3u, Matches.size());
  EXPECT_EQ("6", Matches[1].str());
  EXPECT_EQ("z", Matches[2].str());
  EXPECT_FALSE(r3.match("a6zb6y"));
  EXPECT_FALSE(r3.match("a6zb7z"));
}

TEST_F(RegexTest, Substitution) {
  std::string Error;

  EXPECT_EQ("aNUMber", Regex("[0-9]+").sub("NUM", "a1234ber"));

  // Standard Escapes
  EXPECT_EQ("a\\ber", Regex("[0-9]+").sub("\\\\", "a1234ber", &Error));
  EXPECT_EQ("", Error);
  EXPECT_EQ("a\nber", Regex("[0-9]+").sub("\\n", "a1234ber", &Error));
  EXPECT_EQ("", Error);
  EXPECT_EQ("a\tber", Regex("[0-9]+").sub("\\t", "a1234ber", &Error));
  EXPECT_EQ("", Error);
  EXPECT_EQ("ajber", Regex("[0-9]+").sub("\\j", "a1234ber", &Error));
  EXPECT_EQ("", Error);

  EXPECT_EQ("aber", Regex("[0-9]+").sub("\\", "a1234ber", &Error));
  EXPECT_EQ(Error, "replacement string contained trailing backslash");
  
  // Backreferences
  EXPECT_EQ("aa1234bber", Regex("a[0-9]+b").sub("a\\0b", "a1234ber", &Error));
  EXPECT_EQ("", Error);

  EXPECT_EQ("a1234ber", Regex("a([0-9]+)b").sub("a\\1b", "a1234ber", &Error));
  EXPECT_EQ("", Error);

  EXPECT_EQ("aber", Regex("a[0-9]+b").sub("a\\100b", "a1234ber", &Error));
  EXPECT_EQ(Error, "invalid backreference string '100'");
}

TEST_F(RegexTest, IsLiteralERE) {
  EXPECT_TRUE(Regex::isLiteralERE("abc"));
  EXPECT_FALSE(Regex::isLiteralERE("a(bc)"));
  EXPECT_FALSE(Regex::isLiteralERE("^abc"));
  EXPECT_FALSE(Regex::isLiteralERE("abc$"));
  EXPECT_FALSE(Regex::isLiteralERE("a|bc"));
  EXPECT_FALSE(Regex::isLiteralERE("abc*"));
  EXPECT_FALSE(Regex::isLiteralERE("abc+"));
  EXPECT_FALSE(Regex::isLiteralERE("abc?"));
  EXPECT_FALSE(Regex::isLiteralERE("abc."));
  EXPECT_FALSE(Regex::isLiteralERE("a[bc]"));
  EXPECT_FALSE(Regex::isLiteralERE("abc\\1"));
  EXPECT_FALSE(Regex::isLiteralERE("abc{1,2}"));
}

TEST_F(RegexTest, Escape) {
  EXPECT_EQ("a\\[bc\\]", Regex::escape("a[bc]"));
  EXPECT_EQ("abc\\{1\\\\,2\\}", Regex::escape("abc{1\\,2}"));
}

TEST_F(RegexTest, IsValid) {
  std::string Error;
  EXPECT_FALSE(Regex("(foo").isValid(Error));
  EXPECT_EQ("parentheses not balanced", Error);
  EXPECT_FALSE(Regex("a[b-").isValid(Error));
  EXPECT_EQ("invalid character range", Error);
}

TEST_F(RegexTest, MoveConstruct) {
  Regex r1("^[0-9]+$");
  Regex r2(std::move(r1));
  EXPECT_TRUE(r2.match("916"));
}

TEST_F(RegexTest, MoveAssign) {
  Regex r1("^[0-9]+$");
  Regex r2("abc");
  r2 = std::move(r1);
  EXPECT_TRUE(r2.match("916"));
  std::string Error;
  EXPECT_FALSE(r1.isValid(Error));
}

TEST_F(RegexTest, NoArgConstructor) {
  std::string Error;
  Regex r1;
  EXPECT_FALSE(r1.isValid(Error));
  EXPECT_EQ("invalid regular expression", Error);
  r1 = Regex("abc");
  EXPECT_TRUE(r1.isValid(Error));
}

TEST_F(RegexTest, MatchInvalid) {
  Regex r1;
  std::string Error;
  EXPECT_FALSE(r1.isValid(Error));
  EXPECT_FALSE(r1.match("X"));
}

// https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=3727
TEST_F(RegexTest, OssFuzz3727Regression) {
  // Wrap in a StringRef so the NUL byte doesn't terminate the string
  Regex r(StringRef("[[[=GS\x00[=][", 10));
  std::string Error;
  EXPECT_FALSE(r.isValid(Error));
}

}
