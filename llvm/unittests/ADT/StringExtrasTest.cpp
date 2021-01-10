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

TEST(StringExtrasTest, isSpace) {
  EXPECT_TRUE(isSpace(' '));
  EXPECT_TRUE(isSpace('\t'));
  EXPECT_TRUE(isSpace('\n'));
  EXPECT_TRUE(isSpace('\v'));
  EXPECT_TRUE(isSpace('\f'));
  EXPECT_TRUE(isSpace('\v'));
  EXPECT_FALSE(isSpace('\0'));
  EXPECT_FALSE(isSpace('_'));
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

  std::string InvalidStr = "A5ZX";
  std::string IgnoredOutput;
  EXPECT_FALSE(tryGetFromHex(InvalidStr, IgnoredOutput));
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
  EXPECT_EQ("ABCdef123&<>\\\\\\22'\\09", OS.str());
}

TEST(StringExtrasTest, printHTMLEscaped) {
  std::string str;
  raw_string_ostream OS(str);
  printHTMLEscaped("ABCdef123&<>\"'", OS);
  EXPECT_EQ("ABCdef123&amp;&lt;&gt;&quot;&apos;", OS.str());
}

TEST(StringExtras, ConvertToSnakeFromCamelCase) {
  auto testConvertToSnakeCase = [](llvm::StringRef input,
                                   llvm::StringRef expected) {
    EXPECT_EQ(convertToSnakeFromCamelCase(input), expected.str());
  };

  testConvertToSnakeCase("OpName", "op_name");
  testConvertToSnakeCase("opName", "op_name");
  testConvertToSnakeCase("_OpName", "_op_name");
  testConvertToSnakeCase("Op_Name", "op_name");
  testConvertToSnakeCase("", "");
  testConvertToSnakeCase("A", "a");
  testConvertToSnakeCase("_", "_");
  testConvertToSnakeCase("a", "a");
  testConvertToSnakeCase("op_name", "op_name");
  testConvertToSnakeCase("_op_name", "_op_name");
  testConvertToSnakeCase("__op_name", "__op_name");
  testConvertToSnakeCase("op__name", "op__name");
}

TEST(StringExtras, ConvertToCamelFromSnakeCase) {
  auto testConvertToCamelCase = [](bool capitalizeFirst, llvm::StringRef input,
                                   llvm::StringRef expected) {
    EXPECT_EQ(convertToCamelFromSnakeCase(input, capitalizeFirst),
              expected.str());
  };

  testConvertToCamelCase(false, "op_name", "opName");
  testConvertToCamelCase(false, "_op_name", "_opName");
  testConvertToCamelCase(false, "__op_name", "_OpName");
  testConvertToCamelCase(false, "op__name", "op_Name");
  testConvertToCamelCase(false, "", "");
  testConvertToCamelCase(false, "A", "A");
  testConvertToCamelCase(false, "_", "_");
  testConvertToCamelCase(false, "a", "a");
  testConvertToCamelCase(false, "OpName", "OpName");
  testConvertToCamelCase(false, "opName", "opName");
  testConvertToCamelCase(false, "_OpName", "_OpName");
  testConvertToCamelCase(false, "Op_Name", "Op_Name");
  testConvertToCamelCase(true, "op_name", "OpName");
  testConvertToCamelCase(true, "_op_name", "_opName");
  testConvertToCamelCase(true, "__op_name", "_OpName");
  testConvertToCamelCase(true, "op__name", "Op_Name");
  testConvertToCamelCase(true, "", "");
  testConvertToCamelCase(true, "A", "A");
  testConvertToCamelCase(true, "_", "_");
  testConvertToCamelCase(true, "a", "A");
  testConvertToCamelCase(true, "OpName", "OpName");
  testConvertToCamelCase(true, "_OpName", "_OpName");
  testConvertToCamelCase(true, "Op_Name", "Op_Name");
  testConvertToCamelCase(true, "opName", "OpName");
}

constexpr uint64_t MaxUint64 = std::numeric_limits<uint64_t>::max();
constexpr int64_t MaxInt64 = std::numeric_limits<int64_t>::max();
constexpr int64_t MinInt64 = std::numeric_limits<int64_t>::min();

TEST(StringExtras, UToStr) {
  EXPECT_EQ("0", utostr(0));
  EXPECT_EQ("0", utostr(0, /*isNeg=*/false));
  EXPECT_EQ("1", utostr(1));
  EXPECT_EQ("1", utostr(1, /*isNeg=*/false));
  EXPECT_EQ(std::to_string(MaxUint64), utostr(MaxUint64));
  EXPECT_EQ(std::to_string(MaxUint64), utostr(MaxUint64, /*isNeg=*/false));

  EXPECT_EQ("-0", utostr(0, /*isNeg=*/true));
  EXPECT_EQ("-1", utostr(1, /*isNeg=*/true));
  EXPECT_EQ("-" + std::to_string(MaxInt64), utostr(MaxInt64, /*isNeg=*/true));
  constexpr uint64_t MinusMinInt64 = -static_cast<uint64_t>(MinInt64);
  EXPECT_EQ("-" + std::to_string(MinusMinInt64),
            utostr(MinusMinInt64, /*isNeg=*/true));
  EXPECT_EQ("-" + std::to_string(MaxUint64), utostr(MaxUint64, /*isNeg=*/true));
}

TEST(StringExtras, IToStr) {
  EXPECT_EQ("0", itostr(0));
  EXPECT_EQ("1", itostr(1));
  EXPECT_EQ("-1", itostr(-1));
  EXPECT_EQ(std::to_string(MinInt64), itostr(MinInt64));
  EXPECT_EQ(std::to_string(MaxInt64), itostr(MaxInt64));
}

TEST(StringExtras, SubsequentDelim) {
  SubsequentDelim SD;
  StringRef S = SD;
  EXPECT_EQ(S, "");
  S = SD;
  EXPECT_EQ(S, ", ");

  SubsequentDelim SD2(" ");
  S = SD2;
  EXPECT_EQ(S, "");
  S = SD2;
  EXPECT_EQ(S, " ");
}
