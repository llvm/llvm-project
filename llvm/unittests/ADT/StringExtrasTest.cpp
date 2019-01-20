//===- StringExtrasTest.cpp - Unit tests for String extras ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(StringExtrasTest, isPrint) {
  EXPECT_FALSE(isPrint('\0'));
  EXPECT_FALSE(isPrint('\t'));
  EXPECT_TRUE(isPrint('0'));
  EXPECT_TRUE(isPrint('a'));
  EXPECT_TRUE(isPrint('A'));
  EXPECT_TRUE(isPrint(' '));
  EXPECT_TRUE(isPrint('~'));
  EXPECT_TRUE(isPrint('?'));
}

TEST(StringExtrasTest, Join) {
  std::vector<std::string> Items;
  EXPECT_EQ("", join(Items.begin(), Items.end(), " <sep> "));

  Items = {"foo"};
  EXPECT_EQ("foo", join(Items.begin(), Items.end(), " <sep> "));

  Items = {"foo", "bar"};
  EXPECT_EQ("foo <sep> bar", join(Items.begin(), Items.end(), " <sep> "));

  Items = {"foo", "bar", "baz"};
  EXPECT_EQ("foo <sep> bar <sep> baz",
            join(Items.begin(), Items.end(), " <sep> "));
}

TEST(StringExtrasTest, JoinItems) {
  const char *Foo = "foo";
  std::string Bar = "bar";
  llvm::StringRef Baz = "baz";
  char X = 'x';

  EXPECT_EQ("", join_items(" <sep> "));
  EXPECT_EQ("", join_items('/'));

  EXPECT_EQ("foo", join_items(" <sep> ", Foo));
  EXPECT_EQ("foo", join_items('/', Foo));

  EXPECT_EQ("foo <sep> bar", join_items(" <sep> ", Foo, Bar));
  EXPECT_EQ("foo/bar", join_items('/', Foo, Bar));

  EXPECT_EQ("foo <sep> bar <sep> baz", join_items(" <sep> ", Foo, Bar, Baz));
  EXPECT_EQ("foo/bar/baz", join_items('/', Foo, Bar, Baz));

  EXPECT_EQ("foo <sep> bar <sep> baz <sep> x",
            join_items(" <sep> ", Foo, Bar, Baz, X));

  EXPECT_EQ("foo/bar/baz/x", join_items('/', Foo, Bar, Baz, X));
}

TEST(StringExtrasTest, ToAndFromHex) {
  std::vector<uint8_t> OddBytes = {0x5, 0xBD, 0x0D, 0x3E, 0xCD};
  std::string OddStr = "05BD0D3ECD";
  StringRef OddData(reinterpret_cast<const char *>(OddBytes.data()),
                    OddBytes.size());
  EXPECT_EQ(OddStr, toHex(OddData));
  EXPECT_EQ(OddData, fromHex(StringRef(OddStr).drop_front()));
  EXPECT_EQ(StringRef(OddStr).lower(), toHex(OddData, true));

  std::vector<uint8_t> EvenBytes = {0xA5, 0xBD, 0x0D, 0x3E, 0xCD};
  std::string EvenStr = "A5BD0D3ECD";
  StringRef EvenData(reinterpret_cast<const char *>(EvenBytes.data()),
                     EvenBytes.size());
  EXPECT_EQ(EvenStr, toHex(EvenData));
  EXPECT_EQ(EvenData, fromHex(EvenStr));
  EXPECT_EQ(StringRef(EvenStr).lower(), toHex(EvenData, true));
}

TEST(StringExtrasTest, to_float) {
  float F;
  EXPECT_TRUE(to_float("4.7", F));
  EXPECT_FLOAT_EQ(4.7f, F);

  double D;
  EXPECT_TRUE(to_float("4.7", D));
  EXPECT_DOUBLE_EQ(4.7, D);

  long double LD;
  EXPECT_TRUE(to_float("4.7", LD));
  EXPECT_DOUBLE_EQ(4.7, LD);

  EXPECT_FALSE(to_float("foo", F));
  EXPECT_FALSE(to_float("7.4 foo", F));
  EXPECT_FLOAT_EQ(4.7f, F); // F should be unchanged
}

TEST(StringExtrasTest, printLowerCase) {
  std::string str;
  raw_string_ostream OS(str);
  printLowerCase("ABCdefg01234.,&!~`'}\"", OS);
  EXPECT_EQ("abcdefg01234.,&!~`'}\"", OS.str());
}

TEST(StringExtrasTest, printEscapedString) {
  std::string str;
  raw_string_ostream OS(str);
  printEscapedString("ABCdef123&<>\\\"'\t", OS);
  EXPECT_EQ("ABCdef123&<>\\5C\\22'\\09", OS.str());
}

TEST(StringExtrasTest, printHTMLEscaped) {
  std::string str;
  raw_string_ostream OS(str);
  printHTMLEscaped("ABCdef123&<>\"'", OS);
  EXPECT_EQ("ABCdef123&amp;&lt;&gt;&quot;&apos;", OS.str());
}
