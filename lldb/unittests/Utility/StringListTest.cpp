//===-- StringListTest.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StringList.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(StringListTest, DefaultConstructor) {
  StringList s;
  EXPECT_EQ(0U, s.GetSize());
}

TEST(StringListTest, Assignment) {
  StringList orig;
  orig.AppendString("foo");
  orig.AppendString("bar");

  StringList s = orig;

  ASSERT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));

  ASSERT_EQ(2U, orig.GetSize());
  EXPECT_STREQ("foo", orig.GetStringAtIndex(0));
  EXPECT_STREQ("bar", orig.GetStringAtIndex(1));
}

TEST(StringListTest, AppendStringStdString) {
  StringList s;
  s.AppendString("foo");
  ASSERT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));

  s.AppendString("bar");
  ASSERT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));
}

TEST(StringListTest, AppendStringCString) {
  StringList s;
  s.AppendString("foo", strlen("foo"));
  ASSERT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));

  s.AppendString("bar", strlen("bar"));
  ASSERT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));
}

TEST(StringListTest, AppendStringMove) {
  StringList s;
  std::string foo = "foo";
  std::string bar = "bar";

  s.AppendString(std::move(foo));
  ASSERT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));

  s.AppendString(std::move(bar));
  ASSERT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));
}

TEST(StringListTest, ShiftStdString) {
  StringList s;
  std::string foo = "foo";
  std::string bar = "bar";

  s << foo;
  ASSERT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));

  s << bar;
  ASSERT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));
}

TEST(StringListTest, ShiftCString) {
  StringList s;
  s << "foo";
  ASSERT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));

  s << "bar";
  ASSERT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));
}

TEST(StringListTest, ShiftMove) {
  StringList s;
  std::string foo = "foo";
  std::string bar = "bar";

  s << std::move(foo);
  ASSERT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));

  s << std::move(bar);
  ASSERT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));
}

TEST(StringListTest, AppendListCStringArrayEmpty) {
  StringList s;
  s.AppendList(nullptr, 0);
  EXPECT_EQ(0U, s.GetSize());
}

TEST(StringListTest, AppendListCStringArray) {
  StringList s;
  const char *items[3] = {"foo", "", "bar"};
  s.AppendList(items, 3);

  EXPECT_EQ(3U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("", s.GetStringAtIndex(1));
  EXPECT_STREQ("bar", s.GetStringAtIndex(2));
}

TEST(StringListTest, AppendList) {
  StringList other;
  other.AppendString("foo");
  other.AppendString("");
  other.AppendString("bar");

  StringList empty;

  StringList s;
  s.AppendList(other);

  EXPECT_EQ(3U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("", s.GetStringAtIndex(1));
  EXPECT_STREQ("bar", s.GetStringAtIndex(2));

  EXPECT_EQ(3U, other.GetSize());
  EXPECT_STREQ("foo", other.GetStringAtIndex(0));
  EXPECT_STREQ("", other.GetStringAtIndex(1));
  EXPECT_STREQ("bar", other.GetStringAtIndex(2));

  s.AppendList(empty);
  s.AppendList(other);
  EXPECT_EQ(6U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("", s.GetStringAtIndex(1));
  EXPECT_STREQ("bar", s.GetStringAtIndex(2));
  EXPECT_STREQ("foo", s.GetStringAtIndex(3));
  EXPECT_STREQ("", s.GetStringAtIndex(4));
  EXPECT_STREQ("bar", s.GetStringAtIndex(5));

  EXPECT_EQ(3U, other.GetSize());
  EXPECT_STREQ("foo", other.GetStringAtIndex(0));
  EXPECT_STREQ("", other.GetStringAtIndex(1));
  EXPECT_STREQ("bar", other.GetStringAtIndex(2));
}

TEST(StringListTest, GetSize) {
  StringList s;
  s.AppendString("foo");
  EXPECT_EQ(1U, s.GetSize());

  s.AppendString("foo");
  EXPECT_EQ(2U, s.GetSize());

  s.AppendString("foobar");
  EXPECT_EQ(3U, s.GetSize());
}

TEST(StringListTest, SetSize) {
  StringList s;
  s.SetSize(3);
  EXPECT_EQ(3U, s.GetSize());
  EXPECT_STREQ("", s.GetStringAtIndex(0));
  EXPECT_STREQ("", s.GetStringAtIndex(1));
  EXPECT_STREQ("", s.GetStringAtIndex(2));
}

TEST(StringListTest, SplitIntoLines) {
  StringList s;
  s.SplitIntoLines("\nfoo\nbar\n\n");
  EXPECT_EQ(4U, s.GetSize());
  EXPECT_STREQ("", s.GetStringAtIndex(0));
  EXPECT_STREQ("foo", s.GetStringAtIndex(1));
  EXPECT_STREQ("bar", s.GetStringAtIndex(2));
  EXPECT_STREQ("", s.GetStringAtIndex(3));
}

TEST(StringListTest, SplitIntoLinesSingleTrailingCR) {
  StringList s;
  s.SplitIntoLines("\r");
  EXPECT_EQ(1U, s.GetSize());
  EXPECT_STREQ("", s.GetStringAtIndex(0));
}

TEST(StringListTest, SplitIntoLinesEmpty) {
  StringList s;
  s.SplitIntoLines("");
  EXPECT_EQ(0U, s.GetSize());
}

TEST(StringListTest, LongestCommonPrefixEmpty) {
  StringList s;
  std::string prefix = "this should be cleared";
  s.LongestCommonPrefix(prefix);
  EXPECT_EQ("", prefix);
}

TEST(StringListTest, LongestCommonPrefix) {
  StringList s;
  s.AppendString("foo");
  s.AppendString("foobar");
  s.AppendString("foo");
  s.AppendString("foozar");

  std::string prefix = "this should be cleared";
  s.LongestCommonPrefix(prefix);
  EXPECT_EQ("foo", prefix);
}

TEST(StringListTest, LongestCommonPrefixSingleElement) {
  StringList s;
  s.AppendString("foo");

  std::string prefix = "this should be cleared";
  s.LongestCommonPrefix(prefix);
  EXPECT_EQ("foo", prefix);
}

TEST(StringListTest, LongestCommonPrefixDuplicateElement) {
  StringList s;
  s.AppendString("foo");
  s.AppendString("foo");

  std::string prefix = "this should be cleared";
  s.LongestCommonPrefix(prefix);
  EXPECT_EQ("foo", prefix);
}

TEST(StringListTest, LongestCommonPrefixNoPrefix) {
  StringList s;
  s.AppendString("foo");
  s.AppendString("1foobar");
  s.AppendString("2foo");
  s.AppendString("3foozar");

  std::string prefix = "this should be cleared";
  s.LongestCommonPrefix(prefix);
  EXPECT_EQ("", prefix);
}

TEST(StringListTest, Clear) {
  StringList s;
  s.Clear();
  EXPECT_EQ(0U, s.GetSize());

  s.AppendString("foo");
  s.Clear();
  EXPECT_EQ(0U, s.GetSize());

  s.AppendString("foo");
  s.AppendString("foo");
  s.Clear();
  EXPECT_EQ(0U, s.GetSize());
}

TEST(StringListTest, PopBack) {
  StringList s;
  s.AppendString("foo");
  s.AppendString("bar");
  s.AppendString("boo");

  s.PopBack();
  EXPECT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));

  s.PopBack();
  EXPECT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));

  s.PopBack();
  EXPECT_EQ(0U, s.GetSize());
}

TEST(StringListTest, RemoveBlankLines) {
  StringList s;

  // Nothing to remove yet.
  s.RemoveBlankLines();
  EXPECT_EQ(0U, s.GetSize());

  // Add some lines.
  s.AppendString("");
  s.AppendString("");
  s.AppendString("\t");
  s.AppendString("");
  s.AppendString(" ");
  s.AppendString("");
  s.AppendString("");
  s.AppendString("f");
  s.AppendString("");
  s.AppendString("");

  // And remove all the empty ones again.
  s.RemoveBlankLines();

  EXPECT_EQ(3U, s.GetSize());
  EXPECT_STREQ("\t", s.GetStringAtIndex(0));
  EXPECT_STREQ(" ", s.GetStringAtIndex(1));
  EXPECT_STREQ("f", s.GetStringAtIndex(2));
}

TEST(StringListTest, InsertStringAtIndexStart) {
  StringList s;

  s.InsertStringAtIndex(0, "bar");
  EXPECT_EQ(1U, s.GetSize());
  EXPECT_STREQ("bar", s.GetStringAtIndex(0));

  s.InsertStringAtIndex(0, "foo");
  EXPECT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));
}

TEST(StringListTest, InsertStringAtIndexEnd) {
  StringList s;

  s.InsertStringAtIndex(0, "foo");
  EXPECT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));

  s.InsertStringAtIndex(1, "bar");
  EXPECT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));
}

TEST(StringListTest, InsertStringAtIndexOutOfBounds) {
  StringList s;

  s.InsertStringAtIndex(1, "foo");
  EXPECT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));

  // FIXME: Inserting at an OOB index will always just append to the list. This
  // seems not very intuitive.
  s.InsertStringAtIndex(3, "bar");
  EXPECT_EQ(2U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
  EXPECT_STREQ("bar", s.GetStringAtIndex(1));
}

TEST(StringListTest, InsertStringAtIndexStdString) {
  StringList s;

  std::string foo = "foo";
  s.InsertStringAtIndex(0, foo);
  EXPECT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
}

TEST(StringListTest, InsertStringAtIndexMove) {
  StringList s;

  std::string foo = "foo";
  s.InsertStringAtIndex(0, std::move(foo));
  EXPECT_EQ(1U, s.GetSize());
  EXPECT_STREQ("foo", s.GetStringAtIndex(0));
}

TEST(StringListTest, CopyListEmpty) {
  StringList s;

  EXPECT_EQ("", s.CopyList());
  EXPECT_EQ("", s.CopyList("+"));
}

TEST(StringListTest, CopyListSingle) {
  StringList s;
  s.AppendString("ab");

  EXPECT_EQ("ab", s.CopyList());
  EXPECT_EQ("-ab", s.CopyList("-"));
}

TEST(StringListTest, CopyList) {
  StringList s;
  s.AppendString("ab");
  s.AppendString("cd");

  EXPECT_EQ("ab\ncd", s.CopyList());
  EXPECT_EQ("-ab\n-cd", s.CopyList("-"));
}

TEST(StringListTest, Join) {
  StringList s;
  s.AppendString("ab");
  s.AppendString("cd");

  StreamString ss;
  s.Join(" ", ss);

  EXPECT_EQ("ab cd", ss.GetString());
}

TEST(StringListTest, JoinEmpty) {
  StringList s;

  StreamString ss;
  s.Join(" ", ss);

  EXPECT_EQ("", ss.GetString());
}

TEST(StringListTest, JoinSingle) {
  StringList s;
  s.AppendString("foo");

  StreamString ss;
  s.Join(" ", ss);

  EXPECT_EQ("foo", ss.GetString());
}

TEST(StringListTest, JoinThree) {
  StringList s;
  s.AppendString("1");
  s.AppendString("2");
  s.AppendString("3");

  StreamString ss;
  s.Join(" ", ss);

  EXPECT_EQ("1 2 3", ss.GetString());
}

TEST(StringListTest, JoinNonSpace) {
  StringList s;
  s.AppendString("1");
  s.AppendString("2");
  s.AppendString("3");

  StreamString ss;
  s.Join(".", ss);

  EXPECT_EQ("1.2.3", ss.GetString());
}

TEST(StringListTest, JoinMultiCharSeparator) {
  StringList s;
  s.AppendString("1");
  s.AppendString("2");
  s.AppendString("3");

  StreamString ss;
  s.Join("--", ss);

  EXPECT_EQ("1--2--3", ss.GetString());
}

TEST(StringListTest, GetMaxStringLengthEqualSize) {
  StringList s;
  s.AppendString("123");
  s.AppendString("123");
  EXPECT_EQ(3U, s.GetMaxStringLength());
}

TEST(StringListTest, GetMaxStringLengthIncreasingSize) {
  StringList s;
  s.AppendString("123");
  s.AppendString("1234");
  EXPECT_EQ(4U, s.GetMaxStringLength());
}

TEST(StringListTest, GetMaxStringLengthDecreasingSize) {
  StringList s;
  s.AppendString("1234");
  s.AppendString("123");
  EXPECT_EQ(4U, s.GetMaxStringLength());
}

TEST(StringListTest, GetMaxStringLengthMixed) {
  StringList s;
  s.AppendString("123");
  s.AppendString("1");
  s.AppendString("123");
  s.AppendString("1234");
  s.AppendString("123");
  s.AppendString("1");
  EXPECT_EQ(4U, s.GetMaxStringLength());
}

TEST(StringListTest, GetMaxStringLengthEmpty) {
  StringList s;
  EXPECT_EQ(0U, s.GetMaxStringLength());
}
