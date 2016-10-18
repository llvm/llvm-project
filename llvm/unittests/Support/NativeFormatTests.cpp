//===- llvm/unittest/Support/NativeFormatTests.cpp - formatting tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <type_traits>

using namespace llvm;

namespace {

template <typename T>
std::string format_number(T N, IntegerStyle Style,
                          Optional<size_t> Precision = None,
                          Optional<int> Width = None) {
  std::string S;
  llvm::raw_string_ostream Str(S);
  write_integer(Str, N, Style, Precision, Width);
  Str.flush();
  return S;
}

template <typename T>
typename std::enable_if<std::is_pointer<T>::value, std::string>::type
format_number(T N, HexPrintStyle Style, Optional<size_t> Precision = None,
              Optional<int> Width = None) {
  IntegerStyle IS = hexStyleToIntHexStyle(Style);
  return format_number(reinterpret_cast<uintptr_t>(N), IS, Precision, Width);
}

std::string format_number(uint64_t N, HexPrintStyle Style,
                          Optional<size_t> Precision = None,
                          Optional<int> Width = None) {
  std::string S;
  llvm::raw_string_ostream Str(S);
  write_hex(Str, N, Style, Precision, Width);
  Str.flush();
  return S;
}

std::string format_number(double D, FloatStyle Style,
                          Optional<size_t> Precision = None,
                          Optional<int> Width = None) {
  std::string S;
  llvm::raw_string_ostream Str(S);
  write_double(Str, D, Style, Precision, Width);
  Str.flush();
  return S;
}

// Test basic number formatting with various styles and default width and
// precision.
TEST(NativeFormatTest, BasicIntegerTests) {
  // Simple fixed point integers.  Default precision is 2.
  EXPECT_EQ("0.00", format_number(0, IntegerStyle::Fixed));
  EXPECT_EQ("2425.00", format_number(2425, IntegerStyle::Fixed));
  EXPECT_EQ("-2425.00", format_number(-2425, IntegerStyle::Fixed));

  EXPECT_EQ("0.00", format_number(0LL, IntegerStyle::Fixed));
  EXPECT_EQ("257257257235709.00",
            format_number(257257257235709LL, IntegerStyle::Fixed));
  EXPECT_EQ("-257257257235709.00",
            format_number(-257257257235709LL, IntegerStyle::Fixed));

  // Simple integers with no decimal.  Default precision is 0.
  EXPECT_EQ("0", format_number(0, IntegerStyle::Integer));
  EXPECT_EQ("2425", format_number(2425, IntegerStyle::Integer));
  EXPECT_EQ("-2425", format_number(-2425, IntegerStyle::Integer));

  EXPECT_EQ("0", format_number(0LL, IntegerStyle::Integer));
  EXPECT_EQ("257257257235709",
            format_number(257257257235709LL, IntegerStyle::Integer));
  EXPECT_EQ("-257257257235709",
            format_number(-257257257235709LL, IntegerStyle::Integer));

  // Exponent based integers.  Default precision is 6.
  EXPECT_EQ("3.700000e+01", format_number(37, IntegerStyle::Exponent));
  EXPECT_EQ("4.238000e+03", format_number(4238, IntegerStyle::Exponent));
  EXPECT_EQ("3.700000E+01", format_number(37, IntegerStyle::ExponentUpper));
  EXPECT_EQ("4.238000E+03", format_number(4238, IntegerStyle::ExponentUpper));

  // Number formatting.  Default precision is 0.
  EXPECT_EQ("0", format_number(0, IntegerStyle::Number));
  EXPECT_EQ("2,425", format_number(2425, IntegerStyle::Number));
  EXPECT_EQ("-2,425", format_number(-2425, IntegerStyle::Number));
  EXPECT_EQ("257,257,257,235,709",
            format_number(257257257235709LL, IntegerStyle::Number));
  EXPECT_EQ("-257,257,257,235,709",
            format_number(-257257257235709LL, IntegerStyle::Number));

  // Percent formatting.  Default precision is 0.
  EXPECT_EQ("0%", format_number(0, IntegerStyle::Percent));
  EXPECT_EQ("100%", format_number(1, IntegerStyle::Percent));
  EXPECT_EQ("-100%", format_number(-1, IntegerStyle::Percent));

  // Hex formatting.  Default precision is 0.
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

  EXPECT_EQ("0xFFFFFFFF", format_number(-1, IntegerStyle::HexUpperPrefix));
}

// Test pointer type formatting with various styles and default width and
// precision.
TEST(NativeFormatTest, BasicPointerTests) {
  // lower-case, prefix
  EXPECT_EQ("0x0", format_number((void *)nullptr, HexPrintStyle::PrefixLower));
  EXPECT_EQ("0xbeef",
            format_number((void *)0xbeefLL, HexPrintStyle::PrefixLower));
  EXPECT_EQ("0xdeadbeef",
            format_number((void *)0xdeadbeefLL, HexPrintStyle::PrefixLower));

  // upper-case, prefix.
  EXPECT_EQ("0x0", format_number((void *)nullptr, HexPrintStyle::PrefixUpper));
  EXPECT_EQ("0xBEEF",
            format_number((void *)0xbeefLL, HexPrintStyle::PrefixUpper));
  EXPECT_EQ("0xDEADBEEF",
            format_number((void *)0xdeadbeefLL, HexPrintStyle::PrefixUpper));

  // lower-case, no prefix
  EXPECT_EQ("0", format_number((void *)nullptr, HexPrintStyle::Lower));
  EXPECT_EQ("beef", format_number((void *)0xbeefLL, HexPrintStyle::Lower));
  EXPECT_EQ("deadbeef",
            format_number((void *)0xdeadbeefLL, HexPrintStyle::Lower));

  // upper-case, no prefix.
  EXPECT_EQ("0", format_number((void *)nullptr, HexPrintStyle::Upper));
  EXPECT_EQ("BEEF", format_number((void *)0xbeefLL, HexPrintStyle::Upper));
  EXPECT_EQ("DEADBEEF",
            format_number((void *)0xdeadbeefLL, HexPrintStyle::Upper));
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
}

TEST(NativeFormatTest, HexTests) {
  // Test hex formatting with different widths and precisions.

  // Precision less than the value should print the full value anyway.
  EXPECT_EQ("0x0", format_number(0, IntegerStyle::HexLowerPrefix, 0));
  EXPECT_EQ("0xabcde", format_number(0xABCDE, IntegerStyle::HexLowerPrefix, 3));

  // Precision greater than the value should pad with 0s.
  // TODO: The prefix should not be counted in the precision.  But unfortunately
  // it is and we have to live with it unless we fix all existing users of
  // prefixed hex formatting.
  EXPECT_EQ("0x000", format_number(0, IntegerStyle::HexLowerPrefix, 5));
  EXPECT_EQ("0x0abcde",
            format_number(0xABCDE, IntegerStyle::HexLowerPrefix, 8));

  EXPECT_EQ("00000", format_number(0, IntegerStyle::HexLowerNoPrefix, 5));
  EXPECT_EQ("000abcde",
            format_number(0xABCDE, IntegerStyle::HexLowerNoPrefix, 8));

  // Try printing more digits than can fit in a uint64.
  EXPECT_EQ("0x00000000000000abcde",
            format_number(0xABCDE, IntegerStyle::HexLowerPrefix, 21));

  // Width less than the amount to be printed should print the full amount.
  EXPECT_EQ("0x0", format_number(0, IntegerStyle::HexLowerPrefix, 0, 0));
  EXPECT_EQ("0xabcde",
            format_number(0xABCDE, IntegerStyle::HexLowerPrefix, 0, 0));

  // Width greater than the value should pad with spaces.
  EXPECT_EQ("  0x0", format_number(0, IntegerStyle::HexLowerPrefix, 0, 5));
  EXPECT_EQ(" 0xabcde",
            format_number(0xABCDE, IntegerStyle::HexLowerPrefix, 0, 8));

  // Should also work with no prefix.
  EXPECT_EQ("  000", format_number(0, IntegerStyle::HexLowerNoPrefix, 3, 5));
  EXPECT_EQ("   0abcde",
            format_number(0xABCDE, IntegerStyle::HexLowerNoPrefix, 6, 9));

  // And with pointers.
  EXPECT_EQ("  0x000",
            format_number((void *)nullptr, HexPrintStyle::PrefixLower, 5, 7));

  // Try printing more digits than can fit in a uint64.
  EXPECT_EQ("     0x000abcde",
            format_number(0xABCDE, IntegerStyle::HexLowerPrefix, 10, 15));
}

TEST(NativeFormatTest, IntegerTests) {
  // Test plain integer formatting with non-default widths and precisions.

  // Too low precision should print the whole number.
  EXPECT_EQ("-10", format_number(-10, IntegerStyle::Integer, 1));

  // Additional precision should padd with 0s.
  EXPECT_EQ("-00010", format_number(-10, IntegerStyle::Integer, 5));
  EXPECT_EQ("-00100", format_number(-100, IntegerStyle::Integer, 5));
  EXPECT_EQ("-01000", format_number(-1000, IntegerStyle::Integer, 5));
  EXPECT_EQ("-001234567890",
            format_number(-1234567890, IntegerStyle::Integer, 12));
  EXPECT_EQ("00010", format_number(10, IntegerStyle::Integer, 5));
  EXPECT_EQ("00100", format_number(100, IntegerStyle::Integer, 5));
  EXPECT_EQ("01000", format_number(1000, IntegerStyle::Integer, 5));
  EXPECT_EQ("001234567890",
            format_number(1234567890, IntegerStyle::Integer, 12));

  // Too low width should print the full number.
  EXPECT_EQ("-10", format_number(-10, IntegerStyle::Integer, None, 2));

  // Additional width should padd with spaces.
  EXPECT_EQ("  -00010", format_number(-10, IntegerStyle::Integer, 5, 8));
  EXPECT_EQ("  -00100", format_number(-100, IntegerStyle::Integer, 5, 8));
  EXPECT_EQ("  -01000", format_number(-1000, IntegerStyle::Integer, 5, 8));
  EXPECT_EQ(" -001234567890",
            format_number(-1234567890, IntegerStyle::Integer, 12, 14));
  EXPECT_EQ("   00010", format_number(10, IntegerStyle::Integer, 5, 8));
  EXPECT_EQ("   00100", format_number(100, IntegerStyle::Integer, 5, 8));
  EXPECT_EQ("   01000", format_number(1000, IntegerStyle::Integer, 5, 8));
  EXPECT_EQ("  001234567890",
            format_number(1234567890, IntegerStyle::Integer, 12, 14));
}

TEST(NativeFormatTest, CommaTests) {
  // Test comma grouping with default widths and precisions.
  EXPECT_EQ("0", format_number(0, IntegerStyle::Number));
  EXPECT_EQ("10", format_number(10, IntegerStyle::Number));
  EXPECT_EQ("100", format_number(100, IntegerStyle::Number));
  EXPECT_EQ("1,000", format_number(1000, IntegerStyle::Number));
  EXPECT_EQ("1,234,567,890", format_number(1234567890, IntegerStyle::Number));

  // Test comma grouping with non-default widths and precisions.
  EXPECT_EQ("-10", format_number(-10, IntegerStyle::Number));
  EXPECT_EQ("-100", format_number(-100, IntegerStyle::Number));
  EXPECT_EQ("-1,000", format_number(-1000, IntegerStyle::Number));
  EXPECT_EQ("-1,234,567,890", format_number(-1234567890, IntegerStyle::Number));

  EXPECT_EQ("  1,000", format_number(1000, IntegerStyle::Number, None, 7));
  EXPECT_EQ(" -1,000", format_number(-1000, IntegerStyle::Number, None, 7));
  EXPECT_EQ(" -0,001,000", format_number(-1000, IntegerStyle::Number, 7, 11));
  EXPECT_EQ("  0,001,000", format_number(1000, IntegerStyle::Number, 7, 11));
}

TEST(NativeFormatTest, PercentTests) {
  // Integer percents.
  EXPECT_EQ("0%", format_number(0, IntegerStyle::Percent));
  EXPECT_EQ("0.00%", format_number(0, IntegerStyle::Percent, 2));
  EXPECT_EQ("  0.00%", format_number(0, IntegerStyle::Percent, 2, 7));

  EXPECT_EQ(" 100.00%", format_number(1, IntegerStyle::Percent, 2, 8));

  EXPECT_EQ("    100%", format_number(1, IntegerStyle::Percent, None, 8));
  EXPECT_EQ(" 100.000%", format_number(1, IntegerStyle::Percent, 3, 9));

  // Floating point percents.  Default precision is 2 for floating point types,
  // even for 0.
  EXPECT_EQ("0.00%", format_number(0.0, FloatStyle::Percent));
  EXPECT_EQ("0%", format_number(0.0, FloatStyle::Percent, 0));
  EXPECT_EQ(" 0.00%", format_number(0.0, FloatStyle::Percent, 2, 6));
  EXPECT_EQ(" 4.2%", format_number(.042379, FloatStyle::Percent, 1, 5));
  EXPECT_EQ("4.24%", format_number(.042379, FloatStyle::Percent, 2, 5));
  EXPECT_EQ("4.238%", format_number(.042379, FloatStyle::Percent, 3, 5));
  EXPECT_EQ("  0.424%", format_number(.0042379, FloatStyle::Percent, 3, 8));
  EXPECT_EQ(" -0.424%", format_number(-.0042379, FloatStyle::Percent, 3, 8));
}

TEST(NativeFormatTest, FixedTests) {
  // Integer fixed numbers.  Default precision is 2.  Make sure no decimal
  // is printed with 0 precision.
  EXPECT_EQ("1.00", format_number(1, IntegerStyle::Fixed));
  EXPECT_EQ("1", format_number(1, IntegerStyle::Fixed, 0));
  EXPECT_EQ("  1.00", format_number(1, IntegerStyle::Fixed, 2, 6));
  EXPECT_EQ("-1.00", format_number(-1, IntegerStyle::Fixed));
  EXPECT_EQ("-1.00", format_number(-1, IntegerStyle::Fixed, 2));
  EXPECT_EQ(" -1.00", format_number(-1, IntegerStyle::Fixed, 2, 6));

  // Float fixed numbers.  Default precision is 2.
  EXPECT_EQ("0.00", format_number(0.0, FloatStyle::Fixed));
  EXPECT_EQ("1.00", format_number(1.0, FloatStyle::Fixed));

  // But can be forced to 0
  EXPECT_EQ("0", format_number(0.0, FloatStyle::Fixed, 0));

  // It should round up when appropriate.
  EXPECT_EQ("3.14", format_number(3.1415, FloatStyle::Fixed, 2));
  EXPECT_EQ("3.142", format_number(3.1415, FloatStyle::Fixed, 3));

  // Padding should work properly with both positive and negative numbers.
  EXPECT_EQ("   3.14", format_number(3.1415, FloatStyle::Fixed, 2, 7));
  EXPECT_EQ("  3.142", format_number(3.1415, FloatStyle::Fixed, 3, 7));
  EXPECT_EQ("  -3.14", format_number(-3.1415, FloatStyle::Fixed, 2, 7));
  EXPECT_EQ(" -3.142", format_number(-3.1415, FloatStyle::Fixed, 3, 7));
}
}
