//===-- ArgsTest.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Interpreter/Args.h"

using namespace lldb_private;

TEST(ArgsTest, TestSingleArg) {
  Args args;
  args.SetCommandString("arg");
  EXPECT_EQ(1u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "arg");
}

TEST(ArgsTest, TestSingleQuotedArgWithSpace) {
  Args args;
  args.SetCommandString("\"arg with space\"");
  EXPECT_EQ(1u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "arg with space");
}

TEST(ArgsTest, TestSingleArgWithQuotedSpace) {
  Args args;
  args.SetCommandString("arg\\ with\\ space");
  EXPECT_EQ(1u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "arg with space");
}

TEST(ArgsTest, TestMultipleArgs) {
  Args args;
  args.SetCommandString("this has multiple args");
  EXPECT_EQ(4u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "this");
  EXPECT_STREQ(args.GetArgumentAtIndex(1), "has");
  EXPECT_STREQ(args.GetArgumentAtIndex(2), "multiple");
  EXPECT_STREQ(args.GetArgumentAtIndex(3), "args");
}

TEST(ArgsTest, TestOverwriteArgs) {
  Args args;
  args.SetCommandString("this has multiple args");
  EXPECT_EQ(4u, args.GetArgumentCount());
  args.SetCommandString("arg");
  EXPECT_EQ(1u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "arg");
}

TEST(ArgsTest, TestAppendArg) {
  Args args;
  args.SetCommandString("first_arg");
  EXPECT_EQ(1u, args.GetArgumentCount());
  args.AppendArgument("second_arg");
  EXPECT_EQ(2u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "first_arg");
  EXPECT_STREQ(args.GetArgumentAtIndex(1), "second_arg");
}

TEST(ArgsTest, StringToBoolean) {
  bool success = false;
  EXPECT_TRUE(Args::StringToBoolean("true", false, nullptr));
  EXPECT_TRUE(Args::StringToBoolean("on", false, nullptr));
  EXPECT_TRUE(Args::StringToBoolean("yes", false, nullptr));
  EXPECT_TRUE(Args::StringToBoolean("1", false, nullptr));

  EXPECT_TRUE(Args::StringToBoolean("true", false, &success));
  EXPECT_TRUE(success);
  EXPECT_TRUE(Args::StringToBoolean("on", false, &success));
  EXPECT_TRUE(success);
  EXPECT_TRUE(Args::StringToBoolean("yes", false, &success));
  EXPECT_TRUE(success);
  EXPECT_TRUE(Args::StringToBoolean("1", false, &success));
  EXPECT_TRUE(success);

  EXPECT_FALSE(Args::StringToBoolean("false", true, nullptr));
  EXPECT_FALSE(Args::StringToBoolean("off", true, nullptr));
  EXPECT_FALSE(Args::StringToBoolean("no", true, nullptr));
  EXPECT_FALSE(Args::StringToBoolean("0", true, nullptr));

  EXPECT_FALSE(Args::StringToBoolean("false", true, &success));
  EXPECT_TRUE(success);
  EXPECT_FALSE(Args::StringToBoolean("off", true, &success));
  EXPECT_TRUE(success);
  EXPECT_FALSE(Args::StringToBoolean("no", true, &success));
  EXPECT_TRUE(success);
  EXPECT_FALSE(Args::StringToBoolean("0", true, &success));
  EXPECT_TRUE(success);

  EXPECT_FALSE(Args::StringToBoolean("10", false, &success));
  EXPECT_FALSE(success);
  EXPECT_TRUE(Args::StringToBoolean("10", true, &success));
  EXPECT_FALSE(success);
  EXPECT_TRUE(Args::StringToBoolean("", true, &success));
  EXPECT_FALSE(success);
  EXPECT_TRUE(Args::StringToBoolean(nullptr, true, &success));
  EXPECT_FALSE(success);
}

TEST(ArgsTest, StringToChar) {
  bool success = false;

  EXPECT_EQ('A', Args::StringToChar("A", 'B', nullptr));
  EXPECT_EQ('B', Args::StringToChar("B", 'A', nullptr));

  EXPECT_EQ('A', Args::StringToChar("A", 'B', &success));
  EXPECT_TRUE(success);
  EXPECT_EQ('B', Args::StringToChar("B", 'A', &success));
  EXPECT_TRUE(success);

  EXPECT_EQ('A', Args::StringToChar("", 'A', &success));
  EXPECT_FALSE(success);
  EXPECT_EQ('A', Args::StringToChar("ABC", 'A', &success));
  EXPECT_FALSE(success);
}

TEST(ArgsTest, StringToScriptLanguage) {
  bool success = false;

  EXPECT_EQ(lldb::eScriptLanguageDefault,
            Args::StringToScriptLanguage("default", lldb::eScriptLanguageNone,
                                         nullptr));
  EXPECT_EQ(lldb::eScriptLanguagePython,
            Args::StringToScriptLanguage("python", lldb::eScriptLanguageNone,
                                         nullptr));
  EXPECT_EQ(lldb::eScriptLanguageNone,
            Args::StringToScriptLanguage("none", lldb::eScriptLanguagePython,
                                         nullptr));

  EXPECT_EQ(lldb::eScriptLanguageDefault,
            Args::StringToScriptLanguage("default", lldb::eScriptLanguageNone,
                                         &success));
  EXPECT_TRUE(success);
  EXPECT_EQ(lldb::eScriptLanguagePython,
            Args::StringToScriptLanguage("python", lldb::eScriptLanguageNone,
                                         &success));
  EXPECT_TRUE(success);
  EXPECT_EQ(lldb::eScriptLanguageNone,
            Args::StringToScriptLanguage("none", lldb::eScriptLanguagePython,
                                         &success));
  EXPECT_TRUE(success);

  EXPECT_EQ(lldb::eScriptLanguagePython,
            Args::StringToScriptLanguage("invalid", lldb::eScriptLanguagePython,
                                         &success));
  EXPECT_FALSE(success);
}

TEST(ArgsTest, StringToVersion) {}
