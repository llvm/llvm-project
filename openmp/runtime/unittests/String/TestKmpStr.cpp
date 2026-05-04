//===- TestKmpStr.cpp - Tests for kmp_str utilities ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp_str.h"
#include "gtest/gtest.h"
#include <cstring>

namespace {

// Test basic string buffer initialization
TEST(KmpStrTest, BufferInit) {
  kmp_str_buf_t buffer;
  __kmp_str_buf_init(&buffer);

  EXPECT_NE(buffer.str, nullptr);
  EXPECT_GT(buffer.size, 0u);
  EXPECT_EQ(buffer.used, 0);
  EXPECT_EQ(buffer.str[0], '\0');
}

// Test string buffer clear
TEST(KmpStrTest, BufferClear) {
  kmp_str_buf_t buffer;
  __kmp_str_buf_init(&buffer);
  __kmp_str_buf_print(&buffer, "test string");

  EXPECT_GT(buffer.used, 0);

  __kmp_str_buf_clear(&buffer);
  EXPECT_EQ(buffer.used, 0);
  EXPECT_EQ(buffer.str[0], '\0');

  __kmp_str_buf_free(&buffer);
}

// Test string buffer print
TEST(KmpStrTest, BufferPrint) {
  kmp_str_buf_t buffer;
  __kmp_str_buf_init(&buffer);

  __kmp_str_buf_print(&buffer, "Hello, %s!", "World");

  EXPECT_STREQ(buffer.str, "Hello, World!");
  EXPECT_EQ(buffer.used, 13);

  __kmp_str_buf_free(&buffer);
}

// Test string buffer concatenation
TEST(KmpStrTest, BufferCat) {
  kmp_str_buf_t buffer;
  __kmp_str_buf_init(&buffer);

  __kmp_str_buf_cat(&buffer, "Hello", 5);
  __kmp_str_buf_cat(&buffer, " ", 1);
  __kmp_str_buf_cat(&buffer, "World", 5);

  EXPECT_STREQ(buffer.str, "Hello World");

  __kmp_str_buf_free(&buffer);
}

// Test string buffer reservation
TEST(KmpStrTest, BufferReserve) {
  kmp_str_buf_t buffer;
  __kmp_str_buf_init(&buffer);

  size_t large_size = 2048;
  __kmp_str_buf_reserve(&buffer, large_size);

  EXPECT_GE(buffer.size, large_size);

  __kmp_str_buf_free(&buffer);
}

// Test basic string to int conversion
TEST(KmpStrTest, BasicStrToInt) {
  EXPECT_EQ(__kmp_basic_str_to_int("0"), 0);
  EXPECT_EQ(__kmp_basic_str_to_int("1"), 1);
  EXPECT_EQ(__kmp_basic_str_to_int("42"), 42);
  EXPECT_EQ(__kmp_basic_str_to_int("123"), 123);
}

// Test basic string to int conversion with invalid inputs
TEST(KmpStrTest, BasicStrToIntInvalid) {
  // Empty string returns 0
  EXPECT_EQ(__kmp_basic_str_to_int(""), 0);

  // Strings starting with non-digits return 0
  EXPECT_EQ(__kmp_basic_str_to_int("abc"), 0);
  EXPECT_EQ(__kmp_basic_str_to_int("hello"), 0);
  EXPECT_EQ(__kmp_basic_str_to_int("xyz123"), 0);

  // Special characters return 0
  EXPECT_EQ(__kmp_basic_str_to_int("!@#"), 0);
  EXPECT_EQ(__kmp_basic_str_to_int("+42"), 0);
  EXPECT_EQ(__kmp_basic_str_to_int("-42"), 0);

  // Leading whitespace causes early stop (returns 0)
  EXPECT_EQ(__kmp_basic_str_to_int(" 42"), 0);
  EXPECT_EQ(__kmp_basic_str_to_int("\t42"), 0);

  // Mixed content: parses digits until first non-digit
  EXPECT_EQ(__kmp_basic_str_to_int("123abc"), 123);
  EXPECT_EQ(__kmp_basic_str_to_int("42 "), 42);
  EXPECT_EQ(__kmp_basic_str_to_int("7.5"), 7);
}

// Test string match
TEST(KmpStrTest, StrMatch) {
  const char *data = "Hello World";

  // Test exact match (len == 0)
  EXPECT_TRUE(__kmp_str_match("Hello World", 0, data));
  EXPECT_FALSE(__kmp_str_match("Hello", 0, data)); // Not exact (data is longer)

  // Test prefix match (len < 0)
  EXPECT_TRUE(
      __kmp_str_match("Hello", -1, data)); // "Hello" is prefix of "Hello World"
  EXPECT_FALSE(__kmp_str_match("World", -1, data)); // "World" is not a prefix

  // Test minimum length match (len > 0)
  EXPECT_TRUE(__kmp_str_match("Hello", 5, data)); // At least 5 chars match
  EXPECT_TRUE(__kmp_str_match("Hello", 3, data)); // At least 3 chars match
  EXPECT_FALSE(__kmp_str_match("World", 5, data)); // First chars don't match
}

// Test string contains
TEST(KmpStrTest, StrContains) {
  const char *data = "Hello World";

  EXPECT_TRUE(__kmp_str_contains("Hello", 5, data));
  EXPECT_TRUE(__kmp_str_contains("World", 5, data));
  EXPECT_TRUE(__kmp_str_contains("lo Wo", 5, data));
  EXPECT_FALSE(__kmp_str_contains("Goodbye", 7, data));
}

// Test string match for true/false values
TEST(KmpStrTest, MatchBool) {
  // Test true values
  EXPECT_TRUE(__kmp_str_match_true("true"));
  EXPECT_TRUE(__kmp_str_match_true("TRUE"));
  EXPECT_TRUE(__kmp_str_match_true("on"));
  EXPECT_TRUE(__kmp_str_match_true("ON"));
  EXPECT_TRUE(__kmp_str_match_true("1"));
  EXPECT_TRUE(__kmp_str_match_true("yes"));
  EXPECT_TRUE(__kmp_str_match_true("YES"));

  // Test false values
  EXPECT_TRUE(__kmp_str_match_false("false"));
  EXPECT_TRUE(__kmp_str_match_false("FALSE"));
  EXPECT_TRUE(__kmp_str_match_false("off"));
  EXPECT_TRUE(__kmp_str_match_false("OFF"));
  EXPECT_TRUE(__kmp_str_match_false("0"));
  EXPECT_TRUE(__kmp_str_match_false("no"));
  EXPECT_TRUE(__kmp_str_match_false("NO"));

  // Note: Trailing characters after a valid prefix still match due to
  // minimum-length prefix matching (e.g., "true" uses len=1, "yes" uses len=1)
  EXPECT_TRUE(__kmp_str_match_true("true "));
  EXPECT_TRUE(__kmp_str_match_false("false "));
  EXPECT_TRUE(__kmp_str_match_true("truex"));
  EXPECT_TRUE(__kmp_str_match_false("falsex"));

  // Partial prefixes also match due to minimum-length matching
  EXPECT_TRUE(__kmp_str_match_true("t"));
  EXPECT_TRUE(__kmp_str_match_true("tru"));
  EXPECT_TRUE(__kmp_str_match_false("f"));
  EXPECT_TRUE(__kmp_str_match_false("fals"));
  EXPECT_TRUE(__kmp_str_match_true("y"));
  EXPECT_TRUE(__kmp_str_match_true("yess"));
  EXPECT_TRUE(__kmp_str_match_false("n"));
  EXPECT_TRUE(__kmp_str_match_false("noo"));

  // "on" and "off" require at least 2 characters
  EXPECT_TRUE(__kmp_str_match_true("on"));
  EXPECT_TRUE(__kmp_str_match_false("of"));
  EXPECT_TRUE(__kmp_str_match_false("off"));

  // "enabled" and "disabled" require exact match (len=0)
  EXPECT_TRUE(__kmp_str_match_true("enabled"));
  EXPECT_TRUE(__kmp_str_match_false("disabled"));
}

// Test string match for invalid bool values
TEST(KmpStrTest, MatchBoolInvalid) {
  // Empty string is neither true nor false
  EXPECT_FALSE(__kmp_str_match_true(""));
  EXPECT_FALSE(__kmp_str_match_false(""));

  // Random strings are neither true nor false
  EXPECT_FALSE(__kmp_str_match_true("hello"));
  EXPECT_FALSE(__kmp_str_match_false("hello"));
  EXPECT_FALSE(__kmp_str_match_true("abc"));
  EXPECT_FALSE(__kmp_str_match_false("abc"));

  // Numbers other than 0/1 are neither true nor false
  EXPECT_FALSE(__kmp_str_match_true("2"));
  EXPECT_FALSE(__kmp_str_match_false("2"));
  EXPECT_FALSE(__kmp_str_match_true("42"));
  EXPECT_FALSE(__kmp_str_match_false("42"));
  EXPECT_FALSE(__kmp_str_match_true("-1"));
  EXPECT_FALSE(__kmp_str_match_false("-1"));

  // Leading whitespace prevents matching
  EXPECT_FALSE(__kmp_str_match_true(" true"));
  EXPECT_FALSE(__kmp_str_match_false(" false"));

  // "on" and "off" require at least 2 characters
  EXPECT_FALSE(__kmp_str_match_true("o"));
  EXPECT_FALSE(__kmp_str_match_false("o"));

  // "enabled" and "disabled" require exact match (len=0)
  EXPECT_FALSE(__kmp_str_match_true("enable"));
  EXPECT_FALSE(__kmp_str_match_false("disable"));

  // True values don't match as false and vice versa
  EXPECT_FALSE(__kmp_str_match_false("true"));
  EXPECT_FALSE(__kmp_str_match_false("1"));
  EXPECT_FALSE(__kmp_str_match_false("yes"));
  EXPECT_FALSE(__kmp_str_match_true("false"));
  EXPECT_FALSE(__kmp_str_match_true("0"));
  EXPECT_FALSE(__kmp_str_match_true("no"));
}

// Test string replace
TEST(KmpStrTest, StrReplace) {
  char str[] = "Hello World";
  __kmp_str_replace(str, ' ', '_');
  EXPECT_STREQ(str, "Hello_World");

  __kmp_str_replace(str, 'o', '0');
  EXPECT_STREQ(str, "Hell0_W0rld");
}

// Test string split
TEST(KmpStrTest, StrSplit) {
  char str[] = "key=value";
  char *head = nullptr;
  char *tail = nullptr;

  __kmp_str_split(str, '=', &head, &tail);

  EXPECT_STREQ(head, "key");
  EXPECT_STREQ(tail, "value");
}

// Test file name parsing
TEST(KmpStrTest, FileNameInit) {
  const char *path = "/path/to/file.txt";
  kmp_str_fname_t fname;
  __kmp_str_fname_init(&fname, path);

  EXPECT_NE(fname.path, nullptr);
  EXPECT_STREQ(fname.path, path);
  EXPECT_NE(fname.base, nullptr);
  EXPECT_STREQ(fname.base, "file.txt");

  __kmp_str_fname_free(&fname);
}

// Test string format
TEST(KmpStrTest, StrFormat) {
  char *result = __kmp_str_format("Number: %d, String: %s", 42, "test");

  EXPECT_NE(result, nullptr);
  EXPECT_STREQ(result, "Number: 42, String: test");

  __kmp_str_free(&result);
  EXPECT_EQ(result, nullptr);
}

// Test string buffer concatenate buffers
TEST(KmpStrTest, BufferCatBuf) {
  kmp_str_buf_t buf1, buf2;
  __kmp_str_buf_init(&buf1);
  __kmp_str_buf_init(&buf2);

  __kmp_str_buf_print(&buf1, "Hello");
  __kmp_str_buf_print(&buf2, " World");

  __kmp_str_buf_catbuf(&buf1, &buf2);

  EXPECT_STREQ(buf1.str, "Hello World");

  __kmp_str_buf_free(&buf1);
  __kmp_str_buf_free(&buf2);
}

// Test size string parsing
TEST(KmpStrTest, StrToSize) {
  size_t result;
  const char *error = nullptr;

  __kmp_str_to_size("100", &result, 1, &error);
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(result, 100u);

  __kmp_str_to_size("1K", &result, 1024, &error);
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(result, 1024u);

  __kmp_str_to_size("2M", &result, 1024, &error);
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(result, 2u * 1024u * 1024u);
}

// Test uint string parsing
TEST(KmpStrTest, StrToUint) {
  kmp_uint64 result;
  const char *error = nullptr;

  __kmp_str_to_uint("0", &result, &error);
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(result, 0u);

  __kmp_str_to_uint("42", &result, &error);
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(result, 42u);

  __kmp_str_to_uint("1234567890", &result, &error);
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(result, 1234567890u);
}

} // namespace
