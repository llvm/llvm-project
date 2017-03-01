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

#include <limits>
#include <sstream>

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
  args.AppendArgument(llvm::StringRef("second_arg"));
  EXPECT_EQ(2u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "first_arg");
  EXPECT_STREQ(args.GetArgumentAtIndex(1), "second_arg");
}

TEST(ArgsTest, TestInsertArg) {
  Args args;
  args.AppendArgument("1");
  args.AppendArgument("2");
  args.AppendArgument("3");
  args.InsertArgumentAtIndex(1, "1.5");
  args.InsertArgumentAtIndex(4, "3.5");

  ASSERT_EQ(5u, args.GetArgumentCount());
  EXPECT_STREQ("1", args.GetArgumentAtIndex(0));
  EXPECT_STREQ("1.5", args.GetArgumentAtIndex(1));
  EXPECT_STREQ("2", args.GetArgumentAtIndex(2));
  EXPECT_STREQ("3", args.GetArgumentAtIndex(3));
  EXPECT_STREQ("3.5", args.GetArgumentAtIndex(4));
}

TEST(ArgsTest, TestArgv) {
  Args args;
  EXPECT_EQ(nullptr, args.GetArgumentVector());

  args.AppendArgument("1");
  EXPECT_NE(nullptr, args.GetArgumentVector()[0]);
  EXPECT_EQ(nullptr, args.GetArgumentVector()[1]);

  args.AppendArgument("2");
  EXPECT_NE(nullptr, args.GetArgumentVector()[0]);
  EXPECT_NE(nullptr, args.GetArgumentVector()[1]);
  EXPECT_EQ(nullptr, args.GetArgumentVector()[2]);

  args.AppendArgument("3");
  EXPECT_NE(nullptr, args.GetArgumentVector()[0]);
  EXPECT_NE(nullptr, args.GetArgumentVector()[1]);
  EXPECT_NE(nullptr, args.GetArgumentVector()[2]);
  EXPECT_EQ(nullptr, args.GetArgumentVector()[3]);

  args.InsertArgumentAtIndex(1, "1.5");
  EXPECT_NE(nullptr, args.GetArgumentVector()[0]);
  EXPECT_NE(nullptr, args.GetArgumentVector()[1]);
  EXPECT_NE(nullptr, args.GetArgumentVector()[2]);
  EXPECT_NE(nullptr, args.GetArgumentVector()[3]);
  EXPECT_EQ(nullptr, args.GetArgumentVector()[4]);

  args.InsertArgumentAtIndex(4, "3.5");
  EXPECT_NE(nullptr, args.GetArgumentVector()[0]);
  EXPECT_NE(nullptr, args.GetArgumentVector()[1]);
  EXPECT_NE(nullptr, args.GetArgumentVector()[2]);
  EXPECT_NE(nullptr, args.GetArgumentVector()[3]);
  EXPECT_NE(nullptr, args.GetArgumentVector()[4]);
  EXPECT_EQ(nullptr, args.GetArgumentVector()[5]);
}

TEST(ArgsTest, GetQuotedCommandString) {
  Args args;
  const char *str = "process launch -o stdout.txt -- \"a b c\"";
  args.SetCommandString(str);

  std::string stdstr;
  ASSERT_TRUE(args.GetQuotedCommandString(stdstr));
  EXPECT_EQ(str, stdstr);
}

TEST(ArgsTest, BareSingleQuote) {
  Args args;
  args.SetCommandString("a\\'b");
  EXPECT_EQ(1u, args.GetArgumentCount());

  EXPECT_STREQ("a'b", args.GetArgumentAtIndex(0));
}

TEST(ArgsTest, DoubleQuotedItem) {
  Args args;
  args.SetCommandString("\"a b c\"");
  EXPECT_EQ(1u, args.GetArgumentCount());

  EXPECT_STREQ("a b c", args.GetArgumentAtIndex(0));
}

TEST(ArgsTest, AppendArguments) {
  Args args;
  const char *argv[] = {"1", "2", nullptr};
  const char *argv2[] = {"3", "4", nullptr};

  args.AppendArguments(argv);
  ASSERT_EQ(2u, args.GetArgumentCount());
  EXPECT_STREQ("1", args.GetArgumentVector()[0]);
  EXPECT_STREQ("2", args.GetArgumentVector()[1]);
  EXPECT_EQ(nullptr, args.GetArgumentVector()[2]);
  EXPECT_STREQ("1", args.GetArgumentAtIndex(0));
  EXPECT_STREQ("2", args.GetArgumentAtIndex(1));

  args.AppendArguments(argv2);
  ASSERT_EQ(4u, args.GetArgumentCount());
  EXPECT_STREQ("1", args.GetArgumentVector()[0]);
  EXPECT_STREQ("2", args.GetArgumentVector()[1]);
  EXPECT_STREQ("3", args.GetArgumentVector()[2]);
  EXPECT_STREQ("4", args.GetArgumentVector()[3]);
  EXPECT_EQ(nullptr, args.GetArgumentVector()[4]);
  EXPECT_STREQ("1", args.GetArgumentAtIndex(0));
  EXPECT_STREQ("2", args.GetArgumentAtIndex(1));
  EXPECT_STREQ("3", args.GetArgumentAtIndex(2));
  EXPECT_STREQ("4", args.GetArgumentAtIndex(3));
}

TEST(ArgsTest, GetArgumentArrayRef) {
  Args args("foo bar");
  auto ref = args.GetArgumentArrayRef();
  ASSERT_EQ(2u, ref.size());
  EXPECT_STREQ("foo", ref[0]);
  EXPECT_STREQ("bar", ref[1]);
}

TEST(ArgsTest, StringToBoolean) {
  bool success = false;
  EXPECT_TRUE(Args::StringToBoolean(llvm::StringRef("true"), false, nullptr));
  EXPECT_TRUE(Args::StringToBoolean(llvm::StringRef("on"), false, nullptr));
  EXPECT_TRUE(Args::StringToBoolean(llvm::StringRef("yes"), false, nullptr));
  EXPECT_TRUE(Args::StringToBoolean(llvm::StringRef("1"), false, nullptr));

  EXPECT_TRUE(Args::StringToBoolean(llvm::StringRef("true"), false, &success));
  EXPECT_TRUE(success);
  EXPECT_TRUE(Args::StringToBoolean(llvm::StringRef("on"), false, &success));
  EXPECT_TRUE(success);
  EXPECT_TRUE(Args::StringToBoolean(llvm::StringRef("yes"), false, &success));
  EXPECT_TRUE(success);
  EXPECT_TRUE(Args::StringToBoolean(llvm::StringRef("1"), false, &success));
  EXPECT_TRUE(success);

  EXPECT_FALSE(Args::StringToBoolean(llvm::StringRef("false"), true, nullptr));
  EXPECT_FALSE(Args::StringToBoolean(llvm::StringRef("off"), true, nullptr));
  EXPECT_FALSE(Args::StringToBoolean(llvm::StringRef("no"), true, nullptr));
  EXPECT_FALSE(Args::StringToBoolean(llvm::StringRef("0"), true, nullptr));

  EXPECT_FALSE(Args::StringToBoolean(llvm::StringRef("false"), true, &success));
  EXPECT_TRUE(success);
  EXPECT_FALSE(Args::StringToBoolean(llvm::StringRef("off"), true, &success));
  EXPECT_TRUE(success);
  EXPECT_FALSE(Args::StringToBoolean(llvm::StringRef("no"), true, &success));
  EXPECT_TRUE(success);
  EXPECT_FALSE(Args::StringToBoolean(llvm::StringRef("0"), true, &success));
  EXPECT_TRUE(success);

  EXPECT_FALSE(Args::StringToBoolean(llvm::StringRef("10"), false, &success));
  EXPECT_FALSE(success);
  EXPECT_TRUE(Args::StringToBoolean(llvm::StringRef("10"), true, &success));
  EXPECT_FALSE(success);
  EXPECT_TRUE(Args::StringToBoolean(llvm::StringRef(""), true, &success));
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
            Args::StringToScriptLanguage(llvm::StringRef("default"),
                                         lldb::eScriptLanguageNone, nullptr));
  EXPECT_EQ(lldb::eScriptLanguagePython,
            Args::StringToScriptLanguage(llvm::StringRef("python"),
                                         lldb::eScriptLanguageNone, nullptr));
  EXPECT_EQ(lldb::eScriptLanguageNone,
            Args::StringToScriptLanguage(llvm::StringRef("none"),
                                         lldb::eScriptLanguagePython, nullptr));

  EXPECT_EQ(lldb::eScriptLanguageDefault,
            Args::StringToScriptLanguage(llvm::StringRef("default"),
                                         lldb::eScriptLanguageNone, &success));
  EXPECT_TRUE(success);
  EXPECT_EQ(lldb::eScriptLanguagePython,
            Args::StringToScriptLanguage(llvm::StringRef("python"),
                                         lldb::eScriptLanguageNone, &success));
  EXPECT_TRUE(success);
  EXPECT_EQ(lldb::eScriptLanguageNone,
            Args::StringToScriptLanguage(llvm::StringRef("none"),
                                         lldb::eScriptLanguagePython,
                                         &success));
  EXPECT_TRUE(success);

  EXPECT_EQ(lldb::eScriptLanguagePython,
            Args::StringToScriptLanguage(llvm::StringRef("invalid"),
                                         lldb::eScriptLanguagePython,
                                         &success));
  EXPECT_FALSE(success);
}

TEST(ArgsTest, StringToVersion) {}

// Environment Variable Tests

class EnvVarFixture: public ::testing::Test {
protected:

    void SetUp() {
        args.AppendArgument(llvm::StringRef("Arg1=foo"));
        args.AppendArgument(llvm::StringRef("Arg2"));
        args.AppendArgument(llvm::StringRef("Arg3=bar"));
    }

    size_t GetIndexForEnvVar(llvm::StringRef envvar_name) {
        size_t argument_index = std::numeric_limits<size_t>::max();
        EXPECT_TRUE(args.ContainsEnvironmentVariable(envvar_name,
                                                     &argument_index));
        EXPECT_LT(argument_index, args.GetArgumentCount());
        return argument_index;
    }

    Args  args;
};


TEST_F(EnvVarFixture, TestContainsEnvironmentVariableNoValue) {
    EXPECT_TRUE(args.ContainsEnvironmentVariable(llvm::StringRef("Arg2")));
}

TEST_F(EnvVarFixture, TestContainsEnvironmentVariableWithValue) {
    EXPECT_TRUE(args.ContainsEnvironmentVariable(llvm::StringRef("Arg3")));
}

TEST_F(EnvVarFixture, TestContainsEnvironmentVariableNonExistentVariable) {
    auto nonexistent_envvar = llvm::StringRef("ThisEnvVarShouldNotExist");
    EXPECT_FALSE(args.ContainsEnvironmentVariable(nonexistent_envvar));
}

TEST_F(EnvVarFixture, TestReplaceEnvironmentVariableInitialNoValueWithNoValue) {
    auto envvar_name = llvm::StringRef("Arg2");
    auto argument_index = GetIndexForEnvVar(envvar_name);

    args.AddOrReplaceEnvironmentVariable(envvar_name, llvm::StringRef(""));
    EXPECT_TRUE(args.ContainsEnvironmentVariable(envvar_name));
    EXPECT_EQ(envvar_name, args.GetArgumentAtIndex(argument_index));
}

TEST_F(EnvVarFixture, TestReplaceEnvironmentVariableInitialNoValueWithValue) {
    auto envvar_name = llvm::StringRef("Arg2");
    auto argument_index = GetIndexForEnvVar(envvar_name);

    auto new_value = llvm::StringRef("NewValue");
    args.AddOrReplaceEnvironmentVariable(envvar_name, new_value);
    EXPECT_TRUE(args.ContainsEnvironmentVariable(envvar_name));

    std::stringstream stream;
    stream << envvar_name.str() << '=' << new_value.str();
    EXPECT_EQ(llvm::StringRef(stream.str()),
              args.GetArgumentAtIndex(argument_index));
}

TEST_F(EnvVarFixture, TestReplaceEnvironmentVariableInitialValueWithNoValue) {
    auto envvar_name = llvm::StringRef("Arg1");
    auto argument_index = GetIndexForEnvVar(envvar_name);

    args.AddOrReplaceEnvironmentVariable(envvar_name, llvm::StringRef(""));
    EXPECT_TRUE(args.ContainsEnvironmentVariable(envvar_name));
    EXPECT_EQ(envvar_name, args.GetArgumentAtIndex(argument_index));
}

TEST_F(EnvVarFixture, TestReplaceEnvironmentVariableInitialValueWithValue) {
    auto envvar_name = llvm::StringRef("Arg1");
    auto argument_index = GetIndexForEnvVar(envvar_name);

    auto new_value = llvm::StringRef("NewValue");
    args.AddOrReplaceEnvironmentVariable(envvar_name, new_value);
    EXPECT_TRUE(args.ContainsEnvironmentVariable(envvar_name));

    std::stringstream stream;
    stream << envvar_name.str() << '=' << new_value.str();
    EXPECT_EQ(llvm::StringRef(stream.str()),
              args.GetArgumentAtIndex(argument_index));
}
