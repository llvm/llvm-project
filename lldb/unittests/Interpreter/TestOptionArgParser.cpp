//===-- ArgsTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "llvm/Support/Error.h"

using namespace lldb_private;

TEST(OptionArgParserTest, toBoolean) {
  bool success = false;
  EXPECT_TRUE(
      OptionArgParser::ToBoolean(llvm::StringRef("true"), false, nullptr));
  EXPECT_TRUE(
      OptionArgParser::ToBoolean(llvm::StringRef("on"), false, nullptr));
  EXPECT_TRUE(
      OptionArgParser::ToBoolean(llvm::StringRef("yes"), false, nullptr));
  EXPECT_TRUE(OptionArgParser::ToBoolean(llvm::StringRef("1"), false, nullptr));

  EXPECT_TRUE(
      OptionArgParser::ToBoolean(llvm::StringRef("true"), false, &success));
  EXPECT_TRUE(success);
  EXPECT_TRUE(
      OptionArgParser::ToBoolean(llvm::StringRef("on"), false, &success));
  EXPECT_TRUE(success);
  EXPECT_TRUE(
      OptionArgParser::ToBoolean(llvm::StringRef("yes"), false, &success));
  EXPECT_TRUE(success);
  EXPECT_TRUE(
      OptionArgParser::ToBoolean(llvm::StringRef("1"), false, &success));
  EXPECT_TRUE(success);

  EXPECT_FALSE(
      OptionArgParser::ToBoolean(llvm::StringRef("false"), true, nullptr));
  EXPECT_FALSE(
      OptionArgParser::ToBoolean(llvm::StringRef("off"), true, nullptr));
  EXPECT_FALSE(
      OptionArgParser::ToBoolean(llvm::StringRef("no"), true, nullptr));
  EXPECT_FALSE(OptionArgParser::ToBoolean(llvm::StringRef("0"), true, nullptr));

  EXPECT_FALSE(
      OptionArgParser::ToBoolean(llvm::StringRef("false"), true, &success));
  EXPECT_TRUE(success);
  EXPECT_FALSE(
      OptionArgParser::ToBoolean(llvm::StringRef("off"), true, &success));
  EXPECT_TRUE(success);
  EXPECT_FALSE(
      OptionArgParser::ToBoolean(llvm::StringRef("no"), true, &success));
  EXPECT_TRUE(success);
  EXPECT_FALSE(
      OptionArgParser::ToBoolean(llvm::StringRef("0"), true, &success));
  EXPECT_TRUE(success);

  EXPECT_FALSE(
      OptionArgParser::ToBoolean(llvm::StringRef("10"), false, &success));
  EXPECT_FALSE(success);
  EXPECT_TRUE(
      OptionArgParser::ToBoolean(llvm::StringRef("10"), true, &success));
  EXPECT_FALSE(success);
  EXPECT_TRUE(OptionArgParser::ToBoolean(llvm::StringRef(""), true, &success));
  EXPECT_FALSE(success);
}

void TestToBooleanWithExpectedBool(llvm::StringRef option_arg,
                                   bool expected_parse_success,
                                   bool expected_result) {
  llvm::Expected<bool> bool_or_error =
      OptionArgParser::ToBoolean(llvm::StringRef("test_option"), option_arg);
  EXPECT_EQ(expected_parse_success, (bool)bool_or_error);
  if (expected_parse_success)
    EXPECT_EQ(expected_result, *bool_or_error);
  else {
    std::string error = llvm::toString(bool_or_error.takeError());
    EXPECT_NE(std::string::npos, error.find("test_option"));
  }
}

TEST(OptionArgParserTest, toBooleanWithExpectedBool) {
  TestToBooleanWithExpectedBool(llvm::StringRef("true"), true, true);
  TestToBooleanWithExpectedBool(llvm::StringRef("on"), true, true);
  TestToBooleanWithExpectedBool(llvm::StringRef("yes"), true, true);
  TestToBooleanWithExpectedBool(llvm::StringRef("1"), true, true);

  TestToBooleanWithExpectedBool(llvm::StringRef("True"), true, true);
  TestToBooleanWithExpectedBool(llvm::StringRef("On"), true, true);
  TestToBooleanWithExpectedBool(llvm::StringRef("Yes"), true, true);

  TestToBooleanWithExpectedBool(llvm::StringRef("false"), true, false);
  TestToBooleanWithExpectedBool(llvm::StringRef("off"), true, false);
  TestToBooleanWithExpectedBool(llvm::StringRef("no"), true, false);
  TestToBooleanWithExpectedBool(llvm::StringRef("0"), true, false);

  TestToBooleanWithExpectedBool(llvm::StringRef("False"), true, false);
  TestToBooleanWithExpectedBool(llvm::StringRef("Off"), true, false);
  TestToBooleanWithExpectedBool(llvm::StringRef("No"), true, false);

  TestToBooleanWithExpectedBool(llvm::StringRef("10"), false,
                                false /* doesn't matter */);
  TestToBooleanWithExpectedBool(llvm::StringRef(""), false,
                                false /* doesn't matter */);
}

TEST(OptionArgParserTest, toChar) {
  bool success = false;

  EXPECT_EQ('A', OptionArgParser::ToChar("A", 'B', nullptr));
  EXPECT_EQ('B', OptionArgParser::ToChar("B", 'A', nullptr));

  EXPECT_EQ('A', OptionArgParser::ToChar("A", 'B', &success));
  EXPECT_TRUE(success);
  EXPECT_EQ('B', OptionArgParser::ToChar("B", 'A', &success));
  EXPECT_TRUE(success);

  EXPECT_EQ('A', OptionArgParser::ToChar("", 'A', &success));
  EXPECT_FALSE(success);
  EXPECT_EQ('A', OptionArgParser::ToChar("ABC", 'A', &success));
  EXPECT_FALSE(success);
}

TEST(OptionArgParserTest, toScriptLanguage) {
  bool success = false;

  EXPECT_EQ(lldb::eScriptLanguageDefault,
            OptionArgParser::ToScriptLanguage(llvm::StringRef("default"),
                                              lldb::eScriptLanguageNone,
                                              nullptr));
  EXPECT_EQ(lldb::eScriptLanguagePython,
            OptionArgParser::ToScriptLanguage(
                llvm::StringRef("python"), lldb::eScriptLanguageNone, nullptr));
  EXPECT_EQ(lldb::eScriptLanguageNone,
            OptionArgParser::ToScriptLanguage(
                llvm::StringRef("none"), lldb::eScriptLanguagePython, nullptr));

  EXPECT_EQ(lldb::eScriptLanguageDefault,
            OptionArgParser::ToScriptLanguage(llvm::StringRef("default"),
                                              lldb::eScriptLanguageNone,
                                              &success));
  EXPECT_TRUE(success);
  EXPECT_EQ(lldb::eScriptLanguagePython,
            OptionArgParser::ToScriptLanguage(llvm::StringRef("python"),
                                              lldb::eScriptLanguageNone,
                                              &success));
  EXPECT_TRUE(success);
  EXPECT_EQ(lldb::eScriptLanguageNone,
            OptionArgParser::ToScriptLanguage(llvm::StringRef("none"),
                                              lldb::eScriptLanguagePython,
                                              &success));
  EXPECT_TRUE(success);

  EXPECT_EQ(lldb::eScriptLanguagePython,
            OptionArgParser::ToScriptLanguage(llvm::StringRef("invalid"),
                                              lldb::eScriptLanguagePython,
                                              &success));
  EXPECT_FALSE(success);
}
