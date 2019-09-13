//===-- ArgsTest.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/Args.h"
#include "lldb/Utility/StringList.h"

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

TEST(ArgsTest, TestTrailingBackslash) {
  Args args;
  args.SetCommandString("arg\\");
  EXPECT_EQ(1u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "arg\\");
}

TEST(ArgsTest, TestQuotedTrailingBackslash) {
  Args args;
  args.SetCommandString("\"arg\\");
  EXPECT_EQ(1u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "arg\\");
}

TEST(ArgsTest, TestUnknownEscape) {
  Args args;
  args.SetCommandString("arg\\y");
  EXPECT_EQ(1u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "arg\\y");
}

TEST(ArgsTest, TestQuotedUnknownEscape) {
  Args args;
  args.SetCommandString("\"arg\\y");
  EXPECT_EQ(1u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "arg\\y");
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

TEST(ArgsTest, StringListConstructor) {
  StringList list;
  list << "foo"
       << "bar"
       << "baz";
  Args args(list);
  ASSERT_EQ(3u, args.GetArgumentCount());
  EXPECT_EQ("foo", args[0].ref());
  EXPECT_EQ("bar", args[1].ref());
  EXPECT_EQ("baz", args[2].ref());
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

TEST(ArgsTest, EscapeLLDBCommandArgument) {
  const std::string foo = "foo'";
  EXPECT_EQ("foo\\'", Args::EscapeLLDBCommandArgument(foo, '\0'));
  EXPECT_EQ("foo'", Args::EscapeLLDBCommandArgument(foo, '\''));
  EXPECT_EQ("foo'", Args::EscapeLLDBCommandArgument(foo, '`'));
  EXPECT_EQ("foo'", Args::EscapeLLDBCommandArgument(foo, '"'));

  const std::string bar = "bar\"";
  EXPECT_EQ("bar\\\"", Args::EscapeLLDBCommandArgument(bar, '\0'));
  EXPECT_EQ("bar\"", Args::EscapeLLDBCommandArgument(bar, '\''));
  EXPECT_EQ("bar\"", Args::EscapeLLDBCommandArgument(bar, '`'));
  EXPECT_EQ("bar\\\"", Args::EscapeLLDBCommandArgument(bar, '"'));

  const std::string baz = "baz`";
  EXPECT_EQ("baz\\`", Args::EscapeLLDBCommandArgument(baz, '\0'));
  EXPECT_EQ("baz`", Args::EscapeLLDBCommandArgument(baz, '\''));
  EXPECT_EQ("baz`", Args::EscapeLLDBCommandArgument(baz, '`'));
  EXPECT_EQ("baz\\`", Args::EscapeLLDBCommandArgument(baz, '"'));

  const std::string quux = "quux\t";
  EXPECT_EQ("quux\\\t", Args::EscapeLLDBCommandArgument(quux, '\0'));
  EXPECT_EQ("quux\t", Args::EscapeLLDBCommandArgument(quux, '\''));
  EXPECT_EQ("quux\t", Args::EscapeLLDBCommandArgument(quux, '`'));
  EXPECT_EQ("quux\t", Args::EscapeLLDBCommandArgument(quux, '"'));
}

TEST(ArgsTest, ReplaceArgumentAtIndexShort) {
  Args args;
  args.SetCommandString("foo ba b");
  args.ReplaceArgumentAtIndex(0, "f");
  EXPECT_EQ(3u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "f");
}

TEST(ArgsTest, ReplaceArgumentAtIndexEqual) {
  Args args;
  args.SetCommandString("foo ba b");
  args.ReplaceArgumentAtIndex(0, "bar");
  EXPECT_EQ(3u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "bar");
}

TEST(ArgsTest, ReplaceArgumentAtIndexLonger) {
  Args args;
  args.SetCommandString("foo ba b");
  args.ReplaceArgumentAtIndex(0, "baar");
  EXPECT_EQ(3u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(0), "baar");
}

TEST(ArgsTest, ReplaceArgumentAtIndexOutOfRange) {
  Args args;
  args.SetCommandString("foo ba b");
  args.ReplaceArgumentAtIndex(3, "baar");
  EXPECT_EQ(3u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(2), "b");
}

TEST(ArgsTest, ReplaceArgumentAtIndexFarOutOfRange) {
  Args args;
  args.SetCommandString("foo ba b");
  args.ReplaceArgumentAtIndex(4, "baar");
  EXPECT_EQ(3u, args.GetArgumentCount());
  EXPECT_STREQ(args.GetArgumentAtIndex(2), "b");
}
