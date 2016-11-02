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
