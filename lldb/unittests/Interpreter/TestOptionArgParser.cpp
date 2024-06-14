//===-- ArgsTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb_private;

void TestToBoolean(llvm::StringRef option_arg, bool fail_value,
                   bool expected_value, bool expected_success) {
  EXPECT_EQ(expected_value,
            OptionArgParser::ToBoolean(option_arg, fail_value, nullptr));

  bool success = false;
  EXPECT_EQ(expected_value,
            OptionArgParser::ToBoolean(option_arg, fail_value, &success));
  EXPECT_EQ(expected_success, success);

  Status status;
  EXPECT_EQ(expected_value,
            OptionArgParser::ToBoolean(llvm::StringRef("--test"), option_arg,
                                       fail_value, status));
  EXPECT_EQ(expected_success, status.Success());
}

TEST(OptionArgParserTest, toBoolean) {
  // "True"-ish values should be successfully parsed and return `true`.
  TestToBoolean(llvm::StringRef("true"), false, true, true);
  TestToBoolean(llvm::StringRef("on"), false, true, true);
  TestToBoolean(llvm::StringRef("yes"), false, true, true);
  TestToBoolean(llvm::StringRef("1"), false, true, true);

  // "False"-ish values should be successfully parsed and return `false`.
  TestToBoolean(llvm::StringRef("false"), true, false, true);
  TestToBoolean(llvm::StringRef("off"), true, false, true);
  TestToBoolean(llvm::StringRef("no"), true, false, true);
  TestToBoolean(llvm::StringRef("0"), true, false, true);

  // Other values should fail the parse and return the given `fail_value`.
  TestToBoolean(llvm::StringRef("10"), false, false, false);
  TestToBoolean(llvm::StringRef("10"), true, true, false);
  TestToBoolean(llvm::StringRef(""), false, false, false);
  TestToBoolean(llvm::StringRef(""), true, true, false);
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
