//===-- OptionsWithRawTest.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/Args.h"
#include "lldb/Utility/StringList.h"

using namespace lldb_private;

TEST(OptionsWithRawTest, EmptyInput) {
  // An empty string is just an empty suffix without any arguments.
  OptionsWithRaw args("");
  ASSERT_FALSE(args.HasArgs());
  ASSERT_STREQ(args.GetRawPart().c_str(), "");
}

TEST(OptionsWithRawTest, SingleWhitespaceInput) {
  // Only whitespace is just a suffix.
  OptionsWithRaw args(" ");
  ASSERT_FALSE(args.HasArgs());
  ASSERT_STREQ(args.GetRawPart().c_str(), " ");
}

TEST(OptionsWithRawTest, WhitespaceInput) {
  // Only whitespace is just a suffix.
  OptionsWithRaw args("  ");
  ASSERT_FALSE(args.HasArgs());
  ASSERT_STREQ(args.GetRawPart().c_str(), "  ");
}

TEST(OptionsWithRawTest, ArgsButNoDelimiter) {
  // This counts as a suffix because there is no -- at the end.
  OptionsWithRaw args("-foo bar");
  ASSERT_FALSE(args.HasArgs());
  ASSERT_STREQ(args.GetRawPart().c_str(), "-foo bar");
}

TEST(OptionsWithRawTest, ArgsButNoLeadingDash) {
  // No leading dash means we have no arguments.
  OptionsWithRaw args("foo bar --");
  ASSERT_FALSE(args.HasArgs());
  ASSERT_STREQ(args.GetRawPart().c_str(), "foo bar --");
}

TEST(OptionsWithRawTest, QuotedSuffix) {
  // We need to have a way to escape the -- to make it usable as an argument.
  OptionsWithRaw args("-foo \"--\" bar");
  ASSERT_FALSE(args.HasArgs());
  ASSERT_STREQ(args.GetRawPart().c_str(), "-foo \"--\" bar");
}

TEST(OptionsWithRawTest, EmptySuffix) {
  // An empty suffix with arguments.
  OptionsWithRaw args("-foo --");
  ASSERT_TRUE(args.HasArgs());
  ASSERT_EQ(args.GetArgString(), "-foo ");
  ASSERT_EQ(args.GetArgStringWithDelimiter(), "-foo --");

  auto ref = args.GetArgs().GetArgumentArrayRef();
  ASSERT_EQ(1u, ref.size());
  EXPECT_STREQ("-foo", ref[0]);

  ASSERT_STREQ(args.GetRawPart().c_str(), "");
}

TEST(OptionsWithRawTest, EmptySuffixSingleWhitespace) {
  // A single whitespace also countas as an empty suffix (because that usually
  // separates the suffix from the double dash.
  OptionsWithRaw args("-foo -- ");
  ASSERT_TRUE(args.HasArgs());
  ASSERT_EQ(args.GetArgString(), "-foo ");
  ASSERT_EQ(args.GetArgStringWithDelimiter(), "-foo -- ");

  auto ref = args.GetArgs().GetArgumentArrayRef();
  ASSERT_EQ(1u, ref.size());
  EXPECT_STREQ("-foo", ref[0]);

  ASSERT_STREQ(args.GetRawPart().c_str(), "");
}

TEST(OptionsWithRawTest, WhitespaceSuffix) {
  // A single whtiespace character as a suffix.
  OptionsWithRaw args("-foo --  ");
  ASSERT_TRUE(args.HasArgs());
  ASSERT_EQ(args.GetArgString(), "-foo ");
  ASSERT_EQ(args.GetArgStringWithDelimiter(), "-foo -- ");

  auto ref = args.GetArgs().GetArgumentArrayRef();
  ASSERT_EQ(1u, ref.size());
  EXPECT_STREQ("-foo", ref[0]);

  ASSERT_STREQ(args.GetRawPart().c_str(), " ");
}

TEST(OptionsWithRawTest, LeadingSpaceArgs) {
  // Whitespace before the first dash needs to be ignored.
  OptionsWithRaw args(" -foo -- bar");
  ASSERT_TRUE(args.HasArgs());
  ASSERT_EQ(args.GetArgString(), " -foo ");
  ASSERT_EQ(args.GetArgStringWithDelimiter(), " -foo -- ");

  auto ref = args.GetArgs().GetArgumentArrayRef();
  ASSERT_EQ(1u, ref.size());
  EXPECT_STREQ("-foo", ref[0]);

  ASSERT_STREQ(args.GetRawPart().c_str(), "bar");
}

TEST(OptionsWithRawTest, SingleWordSuffix) {
  // A single word as a suffix.
  OptionsWithRaw args("-foo -- bar");
  ASSERT_TRUE(args.HasArgs());
  ASSERT_EQ(args.GetArgString(), "-foo ");
  ASSERT_EQ(args.GetArgStringWithDelimiter(), "-foo -- ");

  auto ref = args.GetArgs().GetArgumentArrayRef();
  ASSERT_EQ(1u, ref.size());
  EXPECT_STREQ("-foo", ref[0]);

  ASSERT_STREQ(args.GetRawPart().c_str(), "bar");
}

TEST(OptionsWithRawTest, MultiWordSuffix) {
  // Multiple words as a suffix.
  OptionsWithRaw args("-foo -- bar baz");
  ASSERT_TRUE(args.HasArgs());
  ASSERT_EQ(args.GetArgString(), "-foo ");
  ASSERT_EQ(args.GetArgStringWithDelimiter(), "-foo -- ");

  auto ref = args.GetArgs().GetArgumentArrayRef();
  ASSERT_EQ(1u, ref.size());
  EXPECT_STREQ("-foo", ref[0]);

  ASSERT_STREQ(args.GetRawPart().c_str(), "bar baz");
}

TEST(OptionsWithRawTest, UnterminatedQuote) {
  // A quote character in the suffix shouldn't influence the parsing.
  OptionsWithRaw args("-foo -- bar \" ");
  ASSERT_TRUE(args.HasArgs());
  ASSERT_EQ(args.GetArgString(), "-foo ");
  ASSERT_EQ(args.GetArgStringWithDelimiter(), "-foo -- ");

  auto ref = args.GetArgs().GetArgumentArrayRef();
  ASSERT_EQ(1u, ref.size());
  EXPECT_STREQ("-foo", ref[0]);

  ASSERT_STREQ(args.GetRawPart().c_str(), "bar \" ");
}

TEST(OptionsWithRawTest, TerminatedQuote) {
  // A part of the suffix is quoted, which shouldn't influence the parsing.
  OptionsWithRaw args("-foo -- bar \"a\" ");
  ASSERT_TRUE(args.HasArgs());
  ASSERT_EQ(args.GetArgString(), "-foo ");
  ASSERT_EQ(args.GetArgStringWithDelimiter(), "-foo -- ");

  auto ref = args.GetArgs().GetArgumentArrayRef();
  ASSERT_EQ(1u, ref.size());
  EXPECT_STREQ("-foo", ref[0]);

  ASSERT_STREQ(args.GetRawPart().c_str(), "bar \"a\" ");
}

TEST(OptionsWithRawTest, EmptyArgsOnlySuffix) {
  // Empty argument list, but we have a suffix.
  OptionsWithRaw args("-- bar");
  ASSERT_TRUE(args.HasArgs());
  ASSERT_EQ(args.GetArgString(), "");
  ASSERT_EQ(args.GetArgStringWithDelimiter(), "-- ");

  auto ref = args.GetArgs().GetArgumentArrayRef();
  ASSERT_EQ(0u, ref.size());

  ASSERT_STREQ(args.GetRawPart().c_str(), "bar");
}
