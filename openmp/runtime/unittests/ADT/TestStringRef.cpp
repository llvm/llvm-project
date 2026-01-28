//===- TestStringRef.cpp - Tests for kmp_str_ref class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp_adt.h"
#include "kmp.h"
#include "gtest/gtest.h"
#include <cstring>

namespace {

// Helper to compare kmp_str_ref content with a C string
static bool equals(const kmp_str_ref &s, const char *expected) {
  size_t expected_len = strlen(expected);
  if (s.length() != expected_len)
    return false;
  return memcmp(s.begin(), expected, expected_len) == 0;
}

//===----------------------------------------------------------------------===//
// Construction and Basic Properties
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, ConstructFromCString) {
  kmp_str_ref s("Hello");
  EXPECT_EQ(s.length(), 5u);
  EXPECT_TRUE(equals(s, "Hello"));
}

TEST(kmp_str_ref_test, ConstructFromStringView) {
  kmp_str_ref s(std::string_view("Hello World", 5));
  EXPECT_EQ(s.length(), 5u);
  EXPECT_TRUE(equals(s, "Hello"));
}

TEST(kmp_str_ref_test, ConstructEmpty) {
  kmp_str_ref s("");
  EXPECT_EQ(s.length(), 0u);
  EXPECT_TRUE(s.empty());
}

TEST(kmp_str_ref_test, Length) {
  EXPECT_EQ(kmp_str_ref("").length(), 0u);
  EXPECT_EQ(kmp_str_ref("a").length(), 1u);
  EXPECT_EQ(kmp_str_ref("hello").length(), 5u);
  EXPECT_EQ(kmp_str_ref("hello world").length(), 11u);
}

//===----------------------------------------------------------------------===//
// empty
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, EmptyString) {
  kmp_str_ref s("");
  EXPECT_TRUE(s.empty());
}

TEST(kmp_str_ref_test, NonEmptyString) {
  kmp_str_ref s("hello");
  EXPECT_FALSE(s.empty());
}

TEST(kmp_str_ref_test, EmptyAfterConsumeFront) {
  kmp_str_ref s("hello");
  EXPECT_FALSE(s.empty());

  s.consume_front("hello");

  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.length(), 0u);
}

TEST(kmp_str_ref_test, EmptyAfterDropFront) {
  kmp_str_ref s("abc");
  EXPECT_FALSE(s.empty());

  s.drop_front(3);

  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.length(), 0u);
}

TEST(kmp_str_ref_test, EmptyAfterDropWhile) {
  kmp_str_ref s("12345");
  EXPECT_FALSE(s.empty());

  s.drop_while([](char c) {
    return static_cast<bool>(isdigit(static_cast<unsigned char>(c)));
  });

  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.length(), 0u);
}

TEST(kmp_str_ref_test, EmptyAfterConsumeInteger) {
  kmp_str_ref s("42");
  int value = 0;
  EXPECT_FALSE(s.empty());

  s.consume_integer(value);

  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.length(), 0u);
  EXPECT_EQ(value, 42);
}

TEST(kmp_str_ref_test, NotEmptyAfterPartialConsume) {
  kmp_str_ref s("123abc");
  int value = 0;

  s.consume_integer(value);

  EXPECT_FALSE(s.empty());
  EXPECT_EQ(s.length(), 3u);
  EXPECT_TRUE(equals(s, "abc"));
}

//===----------------------------------------------------------------------===//
// Iterators
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, BeginEnd) {
  kmp_str_ref s("Hello");
  EXPECT_EQ(s.end() - s.begin(), 5);
  EXPECT_EQ(*s.begin(), 'H');
}

TEST(kmp_str_ref_test, RangeBasedFor) {
  kmp_str_ref s("abc");
  std::string result;
  for (char c : s) {
    result += c;
  }
  EXPECT_EQ(result, "abc");
}

//===----------------------------------------------------------------------===//
// Assignment
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, Assignment) {
  kmp_str_ref s1("First");
  kmp_str_ref s2("Second");

  s1 = s2;

  EXPECT_TRUE(equals(s1, "Second"));
  EXPECT_EQ(s1.length(), 6u);
}

TEST(kmp_str_ref_test, SelfAssignment) {
  kmp_str_ref s("Test");
  kmp_str_ref &s_ref = s;
  s = s_ref; // Avoid self-assignment warning
  EXPECT_TRUE(equals(s, "Test"));
  EXPECT_EQ(s.length(), 4u);
}

//===----------------------------------------------------------------------===//
// consume_front
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, ConsumeFrontSuccess) {
  kmp_str_ref s("Hello World");

  EXPECT_TRUE(s.consume_front("Hello"));
  EXPECT_EQ(s.length(), 6u);
  EXPECT_TRUE(equals(s, " World"));
}

TEST(kmp_str_ref_test, ConsumeFrontFailure) {
  kmp_str_ref s("Hello World");

  EXPECT_FALSE(s.consume_front("World"));
  EXPECT_EQ(s.length(), 11u);
  EXPECT_TRUE(equals(s, "Hello World"));
}

TEST(kmp_str_ref_test, ConsumeFrontEmpty) {
  kmp_str_ref s("Hello");

  EXPECT_TRUE(s.consume_front(""));
  EXPECT_EQ(s.length(), 5u);
}

TEST(kmp_str_ref_test, ConsumeFrontTooLong) {
  kmp_str_ref s("Hi");

  EXPECT_FALSE(s.consume_front("Hello"));
  EXPECT_EQ(s.length(), 2u);
}

TEST(kmp_str_ref_test, ConsumeFrontExact) {
  kmp_str_ref s("Hello");

  EXPECT_TRUE(s.consume_front("Hello"));
  EXPECT_EQ(s.length(), 0u);
}

TEST(kmp_str_ref_test, ConsumeFrontMultiple) {
  kmp_str_ref s("prefix:middle:suffix");

  EXPECT_TRUE(s.consume_front("prefix"));
  EXPECT_TRUE(s.consume_front(":"));
  EXPECT_TRUE(s.consume_front("middle"));
  EXPECT_TRUE(s.consume_front(":"));
  EXPECT_TRUE(equals(s, "suffix"));
}

//===----------------------------------------------------------------------===//
// consume_integer
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, ConsumeIntegerSimple) {
  kmp_str_ref s("42");
  int value = 0;

  EXPECT_TRUE(s.consume_integer(value));
  EXPECT_EQ(value, 42);
  EXPECT_EQ(s.length(), 0u);
}

TEST(kmp_str_ref_test, ConsumeIntegerWithTrailing) {
  kmp_str_ref s("123abc");
  int value = 0;

  EXPECT_TRUE(s.consume_integer(value));
  EXPECT_EQ(value, 123);
  EXPECT_TRUE(equals(s, "abc"));
}

TEST(kmp_str_ref_test, ConsumeIntegerZero) {
  kmp_str_ref s("0");
  int value = -1;

  // allow_zero = true by default
  EXPECT_TRUE(s.consume_integer(value));
  EXPECT_EQ(value, 0);
  EXPECT_EQ(s.length(), 0u);
}

TEST(kmp_str_ref_test, ConsumeIntegerZeroNotAllowed) {
  kmp_str_ref s("0rest");
  int value = -1;

  EXPECT_FALSE(s.consume_integer(value, /*allow_zero=*/false));
  // State should be restored on failure
  EXPECT_TRUE(equals(s, "0rest"));
}

TEST(kmp_str_ref_test, ConsumeIntegerNoDigits) {
  kmp_str_ref s("abc");
  int value = -1;

  // No digits to consume, should fail
  EXPECT_FALSE(s.consume_integer(value));
  // String should be unchanged
  EXPECT_TRUE(equals(s, "abc"));
}

TEST(kmp_str_ref_test, ConsumeIntegerEmpty) {
  kmp_str_ref s("");
  int value = -1;

  // Empty string has no digits, should fail
  EXPECT_FALSE(s.consume_integer(value));
}

TEST(kmp_str_ref_test, ConsumeIntegerLeadingZero) {
  kmp_str_ref s("007");
  int value = -1;

  EXPECT_TRUE(s.consume_integer(value));
  EXPECT_EQ(value, 7);
  EXPECT_EQ(s.length(), 0u);
}

TEST(kmp_str_ref_test, ConsumeIntegerNegativeAllowed) {
  kmp_str_ref s("-42rest");
  int value = 0;

  EXPECT_TRUE(s.consume_integer(value, true, true));
  EXPECT_EQ(value, -42);
  EXPECT_TRUE(equals(s, "rest"));
}

TEST(kmp_str_ref_test, ConsumeIntegerNegativeNotAllowed) {
  kmp_str_ref s("-42");
  int value = 0;

  EXPECT_FALSE(s.consume_integer(value, true, false));
  // State should be restored on failure
  EXPECT_TRUE(equals(s, "-42"));
}

TEST(kmp_str_ref_test, ConsumeIntegerMultipleDigits) {
  kmp_str_ref s("1234567890");
  int value = 0;

  EXPECT_TRUE(s.consume_integer(value));
  EXPECT_EQ(value, 1234567890);
}

//===----------------------------------------------------------------------===//
// copy
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, Copy) {
  kmp_str_ref s("Hello");
  char *copied = s.copy();

  EXPECT_NE(copied, nullptr);
  EXPECT_STREQ(copied, "Hello");
  EXPECT_NE(copied, s.begin()); // Different pointer

  KMP_INTERNAL_FREE(copied);
}

TEST(kmp_str_ref_test, CopyEmpty) {
  kmp_str_ref s("");
  char *copied = s.copy();

  EXPECT_NE(copied, nullptr);
  EXPECT_STREQ(copied, "");

  KMP_INTERNAL_FREE(copied);
}

TEST(kmp_str_ref_test, CopySubstring) {
  // Test copying a substring that doesn't have a null terminator at len
  kmp_str_ref full("device-0)rest");
  kmp_str_ref sub = full.take_while([](char c) { return c != ')'; });

  EXPECT_EQ(sub.length(), 8u); // "device-0"

  char *copied = sub.copy();

  EXPECT_NE(copied, nullptr);
  EXPECT_STREQ(copied, "device-0"); // Should NOT include ")"
  EXPECT_EQ(strlen(copied), 8u);

  KMP_INTERNAL_FREE(copied);
}

//===----------------------------------------------------------------------===//
// drop_front
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, DropFront) {
  kmp_str_ref s("Hello World");

  s.drop_front(6);

  EXPECT_EQ(s.length(), 5u);
  EXPECT_TRUE(equals(s, "World"));
}

TEST(kmp_str_ref_test, DropFrontZero) {
  kmp_str_ref s("Hello");

  s.drop_front(0);

  EXPECT_EQ(s.length(), 5u);
  EXPECT_TRUE(equals(s, "Hello"));
}

TEST(kmp_str_ref_test, DropFrontAll) {
  kmp_str_ref s("Hello");

  s.drop_front(5);

  EXPECT_EQ(s.length(), 0u);
}

TEST(kmp_str_ref_test, DropFrontMoreThanLength) {
  kmp_str_ref s("Hi");

  s.drop_front(100);

  EXPECT_EQ(s.length(), 0u);
}

//===----------------------------------------------------------------------===//
// drop_while
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, DropWhileDigits) {
  kmp_str_ref s("123abc");

  s.drop_while([](char c) {
    return static_cast<bool>(isdigit(static_cast<unsigned char>(c)));
  });

  EXPECT_TRUE(equals(s, "abc"));
}

TEST(kmp_str_ref_test, DropWhileSpaces) {
  kmp_str_ref s("   hello");

  s.drop_while([](char c) { return c == ' '; });

  EXPECT_TRUE(equals(s, "hello"));
}

TEST(kmp_str_ref_test, DropWhileNone) {
  kmp_str_ref s("hello");

  s.drop_while([](char c) { return c == ' '; });

  EXPECT_TRUE(equals(s, "hello"));
}

TEST(kmp_str_ref_test, DropWhileAll) {
  kmp_str_ref s("12345");

  s.drop_while([](char c) {
    return static_cast<bool>(isdigit(static_cast<unsigned char>(c)));
  });

  EXPECT_EQ(s.length(), 0u);
}

//===----------------------------------------------------------------------===//
// skip_space
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, SkipSpace) {
  kmp_str_ref s("   hello");

  s.skip_space();

  EXPECT_TRUE(equals(s, "hello"));
}

TEST(kmp_str_ref_test, SkipSpaceNoSpaces) {
  kmp_str_ref s("hello");

  s.skip_space();

  EXPECT_TRUE(equals(s, "hello"));
}

TEST(kmp_str_ref_test, SkipSpaceAllSpaces) {
  kmp_str_ref s("     ");

  s.skip_space();

  EXPECT_EQ(s.length(), 0u);
}

TEST(kmp_str_ref_test, SkipSpaceOnlyLeading) {
  kmp_str_ref s("  hello world  ");

  s.skip_space();

  EXPECT_TRUE(equals(s, "hello world  "));
}

TEST(kmp_str_ref_test, SkipSpaceWithTabs) {
  kmp_str_ref s("\t\n  hello");

  s.skip_space();

  EXPECT_TRUE(equals(s, "hello"));
}

//===----------------------------------------------------------------------===//
// take_while
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, TakeWhileDigits) {
  kmp_str_ref s("123abc");

  kmp_str_ref digits = s.take_while([](char c) {
    return static_cast<bool>(isdigit(static_cast<unsigned char>(c)));
  });

  EXPECT_EQ(digits.length(), 3u);
  EXPECT_TRUE(equals(digits, "123"));
  // Original unchanged
  EXPECT_EQ(s.length(), 6u);
}

TEST(kmp_str_ref_test, TakeWhileAlpha) {
  kmp_str_ref s("hello123");

  kmp_str_ref alpha = s.take_while([](char c) {
    return static_cast<bool>(isalpha(static_cast<unsigned char>(c)));
  });

  EXPECT_EQ(alpha.length(), 5u);
  EXPECT_TRUE(equals(alpha, "hello"));
}

TEST(kmp_str_ref_test, TakeWhileNone) {
  kmp_str_ref s("123abc");

  kmp_str_ref result = s.take_while([](char c) {
    return static_cast<bool>(isalpha(static_cast<unsigned char>(c)));
  });

  EXPECT_EQ(result.length(), 0u);
}

TEST(kmp_str_ref_test, TakeWhileAll) {
  kmp_str_ref s("hello");

  kmp_str_ref result = s.take_while([](char c) {
    return static_cast<bool>(isalpha(static_cast<unsigned char>(c)));
  });

  EXPECT_EQ(result.length(), 5u);
  EXPECT_TRUE(equals(result, "hello"));
}

//===----------------------------------------------------------------------===//
// Integration / Complex Scenarios
//===----------------------------------------------------------------------===//

TEST(kmp_str_ref_test, ParseKeyValuePair) {
  kmp_str_ref s("key=value");

  kmp_str_ref key = s.take_while([](char c) { return c != '='; });
  s.drop_front(key.length());
  s.consume_front("=");

  EXPECT_EQ(key.length(), 3u);
  EXPECT_TRUE(equals(key, "key"));
  EXPECT_TRUE(equals(s, "value"));
}

TEST(kmp_str_ref_test, ParseCommaSeparated) {
  kmp_str_ref s("1,2,3");
  int values[3] = {0, 0, 0};
  int count = 0;

  while (s.length() > 0 && count < 3) {
    s.consume_integer(values[count++]);
    s.consume_front(",");
  }

  EXPECT_EQ(count, 3);
  EXPECT_EQ(values[0], 1);
  EXPECT_EQ(values[1], 2);
  EXPECT_EQ(values[2], 3);
}

TEST(kmp_str_ref_test, ParseWithWhitespace) {
  kmp_str_ref s("  hello  world  ");

  s.skip_space();
  kmp_str_ref word1 = s.take_while([](char c) { return c != ' '; });
  s.drop_front(word1.length());
  s.skip_space();
  kmp_str_ref word2 = s.take_while([](char c) { return c != ' '; });

  EXPECT_EQ(word1.length(), 5u);
  EXPECT_TRUE(equals(word1, "hello"));
  EXPECT_EQ(word2.length(), 5u);
  EXPECT_TRUE(equals(word2, "world"));
}

} // namespace
