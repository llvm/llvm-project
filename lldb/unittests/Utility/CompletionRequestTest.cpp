//===-- CompletionRequestTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/CompletionRequest.h"
using namespace lldb_private;

TEST(CompletionRequest, Constructor) {
  std::string command = "a bad c";
  const unsigned cursor_pos = 3;
  const size_t arg_index = 1;
  StringList matches;
  CompletionResult result;

  CompletionRequest request(command, cursor_pos, result);
  result.GetMatches(matches);

  EXPECT_EQ(request.GetRawLine(), "a b");
  EXPECT_EQ(request.GetRawLineWithUnusedSuffix(), command);
  EXPECT_EQ(request.GetRawCursorPos(), cursor_pos);
  EXPECT_EQ(request.GetCursorIndex(), arg_index);

  EXPECT_EQ(request.GetParsedLine().GetArgumentCount(), 2u);
  EXPECT_EQ(request.GetCursorArgumentPrefix().str(), "b");
}

TEST(CompletionRequest, FakeLastArg) {
  // We insert an empty fake argument into the argument list when the
  // cursor is after a space.
  std::string command = "a bad c ";
  const unsigned cursor_pos = command.size();
  CompletionResult result;

  CompletionRequest request(command, cursor_pos, result);

  EXPECT_EQ(request.GetRawLine(), command);
  EXPECT_EQ(request.GetRawLineWithUnusedSuffix(), command);
  EXPECT_EQ(request.GetRawCursorPos(), cursor_pos);
  EXPECT_EQ(request.GetCursorIndex(), 3U);

  EXPECT_EQ(request.GetParsedLine().GetArgumentCount(), 4U);
  EXPECT_EQ(request.GetCursorArgumentPrefix().str(), "");
}

TEST(CompletionRequest, TryCompleteCurrentArgGood) {
  std::string command = "a bad c";
  StringList matches, descriptions;
  CompletionResult result;

  CompletionRequest request(command, 3, result);
  request.TryCompleteCurrentArg("boo", "car");
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(1U, result.GetResults().size());
  EXPECT_STREQ("boo", matches.GetStringAtIndex(0U));
  EXPECT_EQ(1U, descriptions.GetSize());
  EXPECT_STREQ("car", descriptions.GetStringAtIndex(0U));
}

TEST(CompletionRequest, TryCompleteCurrentArgBad) {
  std::string command = "a bad c";
  CompletionResult result;

  CompletionRequest request(command, 3, result);
  request.TryCompleteCurrentArg("car", "card");

  EXPECT_EQ(0U, result.GetResults().size());
}

TEST(CompletionRequest, TryCompleteCurrentArgMode) {
  std::string command = "a bad c";
  CompletionResult result;

  CompletionRequest request(command, 3, result);
  request.TryCompleteCurrentArg<CompletionMode::Partial>("bar", "bard");

  EXPECT_EQ(1U, result.GetResults().size());
  EXPECT_EQ(CompletionMode::Partial, result.GetResults()[0].GetMode());
}

TEST(CompletionRequest, ShiftArguments) {
  std::string command = "a bad c";
  const unsigned cursor_pos = 3;
  const size_t arg_index = 1;
  StringList matches;
  CompletionResult result;

  CompletionRequest request(command, cursor_pos, result);
  result.GetMatches(matches);

  EXPECT_EQ(request.GetRawLine(), "a b");
  EXPECT_EQ(request.GetRawLineWithUnusedSuffix(), command);
  EXPECT_EQ(request.GetRawCursorPos(), cursor_pos);
  EXPECT_EQ(request.GetCursorIndex(), arg_index);

  EXPECT_EQ(request.GetParsedLine().GetArgumentCount(), 2u);
  EXPECT_STREQ(request.GetParsedLine().GetArgumentAtIndex(1), "b");

  // Shift away the 'a' argument.
  request.ShiftArguments();

  // The raw line/cursor stays identical.
  EXPECT_EQ(request.GetRawLine(), "a b");
  EXPECT_EQ(request.GetRawLineWithUnusedSuffix(), command);
  EXPECT_EQ(request.GetRawCursorPos(), cursor_pos);

  // Partially parsed line and cursor should be updated.
  EXPECT_EQ(request.GetCursorIndex(), arg_index - 1U);
  EXPECT_EQ(request.GetParsedLine().GetArgumentCount(), 1u);
  EXPECT_EQ(request.GetCursorArgumentPrefix().str(), "b");
}

TEST(CompletionRequest, DuplicateFiltering) {
  std::string command = "a bad c";
  const unsigned cursor_pos = 3;
  StringList matches;

  CompletionResult result;
  CompletionRequest request(command, cursor_pos, result);
  result.GetMatches(matches);

  EXPECT_EQ(0U, result.GetNumberOfResults());

  // Add foo twice
  request.AddCompletion("foo");
  result.GetMatches(matches);

  EXPECT_EQ(1U, result.GetNumberOfResults());
  EXPECT_EQ(1U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));

  request.AddCompletion("foo");
  result.GetMatches(matches);

  EXPECT_EQ(1U, result.GetNumberOfResults());
  EXPECT_EQ(1U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));

  // Add bar twice
  request.AddCompletion("bar");
  result.GetMatches(matches);

  EXPECT_EQ(2U, result.GetNumberOfResults());
  EXPECT_EQ(2U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));

  request.AddCompletion("bar");
  result.GetMatches(matches);

  EXPECT_EQ(2U, result.GetNumberOfResults());
  EXPECT_EQ(2U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));

  // Add foo again.
  request.AddCompletion("foo");
  result.GetMatches(matches);

  EXPECT_EQ(2U, result.GetNumberOfResults());
  EXPECT_EQ(2U, matches.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));

  // Add something with an existing prefix
  request.AddCompletion("foobar");
  result.GetMatches(matches);

  EXPECT_EQ(3U, result.GetNumberOfResults());
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
  CompletionRequest request(command, cursor_pos, result);
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(0U, result.GetNumberOfResults());

  // Add foo twice with same comment
  request.AddCompletion("foo", "comment");
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(1U, result.GetNumberOfResults());
  EXPECT_EQ(1U, matches.GetSize());
  EXPECT_EQ(1U, descriptions.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("comment", descriptions.GetStringAtIndex(0));

  request.AddCompletion("foo", "comment");
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(1U, result.GetNumberOfResults());
  EXPECT_EQ(1U, matches.GetSize());
  EXPECT_EQ(1U, descriptions.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("comment", descriptions.GetStringAtIndex(0));

  // Add bar twice with different comments
  request.AddCompletion("bar", "comment");
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(2U, result.GetNumberOfResults());
  EXPECT_EQ(2U, matches.GetSize());
  EXPECT_EQ(2U, descriptions.GetSize());
  EXPECT_STREQ("foo", matches.GetStringAtIndex(0));
  EXPECT_STREQ("bar", matches.GetStringAtIndex(1));

  request.AddCompletion("bar", "another comment");
  result.GetMatches(matches);
  result.GetDescriptions(descriptions);

  EXPECT_EQ(3U, result.GetNumberOfResults());
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

  EXPECT_EQ(4U, result.GetNumberOfResults());
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
  CompletionRequest request(command, cursor_pos, result);

  std::string Temporary = "bar";
  request.AddCompletion(Temporary);
  // Manipulate our completion. The request should have taken a copy, so that
  // shouldn't influence anything.
  Temporary[0] = 'f';

  result.GetMatches(matches);
  EXPECT_EQ(1U, result.GetNumberOfResults());
  EXPECT_STREQ("bar", matches.GetStringAtIndex(0));
}
