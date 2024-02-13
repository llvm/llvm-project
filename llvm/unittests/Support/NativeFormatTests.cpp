//===- llvm/unittest/Support/NativeFormatTests.cpp - formatting tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <type_traits>

using namespace llvm;

namespace {

template <typename T> std::string format_number(T N, IntegerStyle Style) {
  std::string S;
  llvm::raw_string_ostream Str(S);
  write_integer(Str, N, 0, Style);
  Str.flush();
  return S;
}

std::string format_number(uint64_t N, HexPrintStyle Style,
                          std::optional<size_t> Width = std::nullopt) {
  std::string S;
  llvm::raw_string_ostream Str(S);
  write_hex(Str, N, Style, Width);
  Str.flush();
  return S;
}

std::string format_number(double D, FloatStyle Style,
                          std::optional<size_t> Precision = std::nullopt) {
  std::string S;
  llvm::raw_string_ostream Str(S);
  write_double(Str, D, Style, Precision);
  Str.flush();
  return S;
}

// Test basic number formatting with various styles and default width and
// precision.
TEST(NativeFormatTest, BasicIntegerTests) {
  // Simple integers with no decimal.
  EXPECT_EQ("0", format_number(0, IntegerStyle::Integer));
  EXPECT_EQ("1", format_number(1, IntegerStyle::Integer));
  EXPECT_EQ("-1", format_number(-1, IntegerStyle::Integer));
  EXPECT_EQ("2425", format_number(2425, IntegerStyle::Integer));
  EXPECT_EQ("-2425", format_number(-2425, IntegerStyle::Integer));

  EXPECT_EQ("0", format_number(0LL, IntegerStyle::Integer));
  EXPECT_EQ("257257257235709",
            format_number(257257257235709LL, IntegerStyle::Integer));
  EXPECT_EQ("-257257257235709",
            format_number(-257257257235709LL, IntegerStyle::Integer));

  // Number formatting.
  EXPECT_EQ("0", format_number(0, IntegerStyle::Number));
  EXPECT_EQ("2,425", format_number(2425, IntegerStyle::Number));
  EXPECT_EQ("-2,425", format_number(-2425, IntegerStyle::Number));
  EXPECT_EQ("257,257,257,235,709",
            format_number(257257257235709LL, IntegerStyle::Number));
  EXPECT_EQ("-257,257,257,235,709",
            format_number(-257257257235709LL, IntegerStyle::Number));

  // Hex formatting.
  // lower case, prefix.
  EXPECT_EQ("0x0", format_number(0, HexPrintStyle::PrefixLower));
  EXPECT_EQ("0xbeef", format_number(0xbeefLL, HexPrintStyle::PrefixLower));
  EXPECT_EQ("0xdeadbeef",
            format_number(0xdeadbeefLL, HexPrintStyle::PrefixLower));

  // upper-case, prefix.
  EXPECT_EQ("0x0", format_number(0, HexPrintStyle::PrefixUpper));
  EXPECT_EQ("0xBEEF", format_number(0xbeefLL, HexPrintStyle::PrefixUpper));
  EXPECT_EQ("0xDEADBEEF",
            format_number(0xdeadbeefLL, HexPrintStyle::PrefixUpper));

  // lower-case, no prefix
  EXPECT_EQ("0", format_number(0, HexPrintStyle::Lower));
  EXPECT_EQ("beef", format_number(0xbeefLL, HexPrintStyle::Lower));
  EXPECT_EQ("deadbeef", format_number(0xdeadbeefLL, HexPrintStyle::Lower));

  // upper-case, no prefix.
  EXPECT_EQ("0", format_number(0, HexPrintStyle::Upper));
  EXPECT_EQ("BEEF", format_number(0xbeef, HexPrintStyle::Upper));
  EXPECT_EQ("DEADBEEF", format_number(0xdeadbeef, HexPrintStyle::Upper));
}

// Test basic floating point formatting with various styles and default width
// and precision.
TEST(NativeFormatTest, BasicFloatingPointTests) {
  // Double
  EXPECT_EQ("0.000000e+00", format_number(0.0, FloatStyle::Exponent));
  EXPECT_EQ("-0.000000e+00", format_number(-0.0, FloatStyle::Exponent));
  EXPECT_EQ("1.100000e+00", format_number(1.1, FloatStyle::Exponent));
  EXPECT_EQ("1.100000E+00", format_number(1.1, FloatStyle::ExponentUpper));

  // Default precision is 2 for floating points.
  EXPECT_EQ("1.10", format_number(1.1, FloatStyle::Fixed));
  EXPECT_EQ("1.34", format_number(1.34, FloatStyle::Fixed));
  EXPECT_EQ("1.34", format_number(1.344, FloatStyle::Fixed));
  EXPECT_EQ("1.35", format_number(1.346, FloatStyle::Fixed));
}

// Test common boundary cases and min/max conditions.
TEST(NativeFormatTest, BoundaryTests) {
  // Min and max.
  EXPECT_EQ("18446744073709551615",
            format_number(UINT64_MAX, IntegerStyle::Integer));

  EXPECT_EQ("9223372036854775807",
            format_number(INT64_MAX, IntegerStyle::Integer));
  EXPECT_EQ("-9223372036854775808",
            format_number(INT64_MIN, IntegerStyle::Integer));

  EXPECT_EQ("4294967295", format_number(UINT32_MAX, IntegerStyle::Integer));
  EXPECT_EQ("2147483647", format_number(INT32_MAX, IntegerStyle::Integer));
  EXPECT_EQ("-2147483648", format_number(INT32_MIN, IntegerStyle::Integer));

  EXPECT_EQ("nan", format_number(std::numeric_limits<double>::quiet_NaN(),
                                 FloatStyle::Fixed));
  EXPECT_EQ("INF", format_number(std::numeric_limits<double>::infinity(),
                                 FloatStyle::Fixed));
  EXPECT_EQ("-INF", format_number(-std::numeric_limits<double>::infinity(),
                                  FloatStyle::Fixed));
}

TEST(NativeFormatTest, HexTests) {
  // Test hex formatting with different widths and precisions.

  // Width less than the value should print the full value anyway.
  EXPECT_EQ("0x0", format_number(0, HexPrintStyle::PrefixLower, 0));
  EXPECT_EQ("0xabcde", format_number(0xABCDE, HexPrintStyle::PrefixLower, 3));

  // Precision greater than the value should pad with 0s.
  // TODO: The prefix should not be counted in the precision.  But unfortunately
  // it is and we have to live with it unless we fix all existing users of
  // prefixed hex formatting.
  EXPECT_EQ("0x000", format_number(0, HexPrintStyle::PrefixLower, 5));
  EXPECT_EQ("0x0abcde", format_number(0xABCDE, HexPrintStyle::PrefixLower, 8));

  EXPECT_EQ("00000", format_number(0, HexPrintStyle::Lower, 5));
  EXPECT_EQ("000abcde", format_number(0xABCDE, HexPrintStyle::Lower, 8));

  // Try printing more digits than can fit in a uint64.
  EXPECT_EQ("0x00000000000000abcde",
            format_number(0xABCDE, HexPrintStyle::PrefixLower, 21));
}

TEST(NativeFormatTest, IntegerTests) {
  EXPECT_EQ("-10", format_number(-10, IntegerStyle::Integer));
  EXPECT_EQ("-100", format_number(-100, IntegerStyle::Integer));
  EXPECT_EQ("-1000", format_number(-1000, IntegerStyle::Integer));
  EXPECT_EQ("-1234567890", format_number(-1234567890, IntegerStyle::Integer));
  EXPECT_EQ("10", format_number(10, IntegerStyle::Integer));
  EXPECT_EQ("100", format_number(100, IntegerStyle::Integer));
  EXPECT_EQ("1000", format_number(1000, IntegerStyle::Integer));
  EXPECT_EQ("1234567890", format_number(1234567890, IntegerStyle::Integer));
}

TEST(NativeFormatTest, CommaTests) {
  EXPECT_EQ("0", format_number(0, IntegerStyle::Number));
  EXPECT_EQ("10", format_number(10, IntegerStyle::Number));
  EXPECT_EQ("100", format_number(100, IntegerStyle::Number));
  EXPECT_EQ("1,000", format_number(1000, IntegerStyle::Number));
  EXPECT_EQ("1,234,567,890", format_number(1234567890, IntegerStyle::Number));

  EXPECT_EQ("-10", format_number(-10, IntegerStyle::Number));
  EXPECT_EQ("-100", format_number(-100, IntegerStyle::Number));
  EXPECT_EQ("-1,000", format_number(-1000, IntegerStyle::Number));
  EXPECT_EQ("-1,234,567,890", format_number(-1234567890, IntegerStyle::Number));
}
}
