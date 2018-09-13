//===-- CompletionRequestTest.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/CompletionRequest.h"
using namespace lldb_private;

TEST(CompletionRequest, Constructor) {
  std::string command = "a bad c";
  const unsigned cursor_pos = 3;
  const int arg_index = 1;
  const int arg_cursor_pos = 1;
  const int match_start = 2345;
  const int match_max_return = 12345;
  StringList matches;
  CompletionResult result;

  CompletionRequest request(command, cursor_pos, match_start, match_max_return,
                            result);
  result.GetMatches(matches);

  EXPECT_STREQ(request.GetRawLine().str().c_str(), command.c_str());
  EXPECT_EQ(request.GetRawCursorPos(), cursor_pos);
  EXPECT_EQ(request.GetCursorIndex(), arg_index);
  EXPECT_EQ(request.GetCursorCharPosition(), arg_cursor_pos);
  EXPECT_EQ(request.GetMatchStartPoint(), match_start);
  EXPECT_EQ(request.GetMaxReturnElements(), match_max_return);
  EXPECT_EQ(request.GetWordComplete(), false);

  EXPECT_EQ(request.GetPartialParsedLine().GetArgumentCount(), 2u);
  EXPECT_STREQ(request.GetPartialParsedLine().GetArgumentAtIndex(1), "b");
}

TEST(CompletionRequest, DuplicateFiltering) {
  std::string command = "a bad c";
  const unsigned cursor_pos = 3;
  StringList matches;

  CompletionResult result;
  CompletionRequest request(command, cursor_pos, 0, 0, result);
  result.GetMatches(matches);

  EXPECT_EQ(0U, request.GetNumberOfMatches());

  // Add foo twice
  request.AddCompletion("foo");
  result.GetMatches(matches);

  EXPECT_EQ(1U, request.GetNumberOfMatches());
  EXPECT_EQ(1U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));

  request.AddCompletion("foo");
  result.GetMatches(matches);

  EXPECT_EQ(1U, request.GetNumberOfMatches());
  EXPECT_EQ(1U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));

  // Add bar twice
  request.AddCompletion("bar");
  result.GetMatches(matches);

  EXPECT_EQ(2U, request.GetNumberOfMatches());
  EXPECT_EQ(2U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));

  request.AddCompletion("bar");
  result.GetMatches(matches);

  EXPECT_EQ(2U, request.GetNumberOfMatches());
  EXPECT_EQ(2U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));

  // Add foo again.
  request.AddCompletion("foo");
  result.GetMatches(matches);

  EXPECT_EQ(2U, request.GetNumberOfMatches());
  EXPECT_EQ(2U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));

  // Add something with an existing prefix
  request.AddCompletion("foobar");
  result.GetMatches(matches);

  EXPECT_EQ(3U, request.GetNumberOfMatches());
  EXPECT_EQ(3U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));
  EXPECT_STREQ("foobar", matches.GetStringAtIndex(2));
}

TEST(CompletionRequest, DuplicateFilteringWithComments) {
  std::string command = "a bad c";
  const unsigned cursor_pos = 3;
  StringList matches, descriptions;

  CompletionResult result;
  CompletionRequest request(command, cursor_pos, 0, 0, result);
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(0U, request.GetNumberOfMatches());

  // Add foo twice with same comment
  request.AddCompletion("foo", "comment");
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(1U, request.GetNumberOfMatches());
  EXPECT_EQ(1U, matches.GetSize());
  EXPECT_EQ(1U, descriptions.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("comment", descriptions.GetStringAtIndex(0));

  request.AddCompletion("foo", "comment");
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(1U, request.GetNumberOfMatches());
  EXPECT_EQ(1U, matches.GetSize());
  EXPECT_EQ(1U, descriptions.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("comment", descriptions.GetStringAtIndex(0));

  // Add bar twice with different comments
  request.AddCompletion("bar", "comment");
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(2U, request.GetNumberOfMatches());
  EXPECT_EQ(2U, matches.GetSize());
  EXPECT_EQ(2U, descriptions.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));

  request.AddCompletion("bar", "another comment");
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(3U, request.GetNumberOfMatches());
  EXPECT_EQ(3U, matches.GetSize());
  EXPECT_EQ(3U, descriptions.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("comment", descriptions.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));
  EXPECT_STREQ("comment", descriptions.GetStringAtIndex(1));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(2));
  EXPECT_STREQ("another comment", descriptions.GetStringAtIndex(2));

  // Add foo again with no comment
  request.AddCompletion("foo");
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(4U, request.GetNumberOfMatches());
  EXPECT_EQ(4U, matches.GetSize());
  EXPECT_EQ(4U, descriptions.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("comment", descriptions.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));
  EXPECT_STREQ("comment", descriptions.GetStringAtIndex(1));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(2));
  EXPECT_STREQ("another comment", descriptions.GetStringAtIndex(2));
  EXPECT_STREQ("foo", matches.GetStringAtIndex(3));
  EXPECT_STREQ("", descriptions.GetStringAtIndex(3));
}

TEST(CompletionRequest, TestCompletionOwnership) {
  std::string command = "a bad c";
  const unsigned cursor_pos = 3;
  StringList matches;

  CompletionResult result;
  CompletionRequest request(command, cursor_pos, 0, 0, result);

  std::string Temporary = "bar";
  request.AddCompletion(Temporary);
  // Manipulate our completion. The request should have taken a copy, so that
  // shouldn't influence anything.
  Temporary[0] = 'f';

  result.GetMatches(matches);
  EXPECT_EQ(1U, request.GetNumberOfMatches());
  EXPECT_STREQ("bar", matches.GetStringAtIndex(0));
}
