//===- StringExtrasTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for string utility APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/StringExtras.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

using namespace orc_rt;

TEST(StringExtrasTest, JoinSingleString) { EXPECT_EQ(join({"x"}, "-"), "x"); }

TEST(StringExtrasTest, JoinMultipleString) {
  EXPECT_EQ(join({"a", "b", "c"}, "-"), "a-b-c");
}

TEST(StringExtrasTest, JoinMantainsEmptyString) {
  EXPECT_EQ(join({"a", "", "c"}, "-"), "a--c");
  EXPECT_EQ(join({"", "b"}, "-"), "-b");
  EXPECT_EQ(join({"a", ""}, "-"), "a-");
}

TEST(StringExtrasTest, JoinEmptySeparator) {
  EXPECT_EQ(join({"a", "b", "c"}, ""), "abc");
}

TEST(StringExtrasTest, JoinMultiSeparator) {
  EXPECT_EQ(join({"a", "b", "c"}, ", "), "a, b, c");
}

TEST(StringOutputStreamTest, EmptyByDefault) {
  StringOutputStream OS;
  EXPECT_EQ(OS.str(), "");
}

TEST(StringOutputStreamTest, TextTypes) {
  const char *CStr = "cstr";
  std::string Str = "std::string";
  std::string_view SV = "view";
  StringOutputStream OS;
  OS << "lit " << CStr << ' ' << Str << ' ' << SV;
  EXPECT_EQ(OS.str(), "lit cstr std::string view");
}

// char and bool must render as a character / "true"|"false", NOT as numbers.
// This pins the non-template overloads against the integral template.
TEST(StringOutputStreamTest, CharAndBoolAreNotNumeric) {
  StringOutputStream OS;
  OS << 'A' << ':' << true << ',' << false;
  EXPECT_EQ(OS.str(), "A:true,false");
}

TEST(StringOutputStreamTest, IntegersAreDecimalByDefault) {
  StringOutputStream OS;
  OS << 0 << ' ' << 42 << ' ' << -7 << ' ' << 3000000000u;
  EXPECT_EQ(OS.str(), "0 42 -7 3000000000");
}

// Buffer is sized from the type, so the widest built-in integers round-trip.
TEST(StringOutputStreamTest, IntegerExtremes) {
  {
    StringOutputStream OS;
    OS << std::numeric_limits<int64_t>::min();
    EXPECT_EQ(OS.str(), "-9223372036854775808");
  }
  {
    StringOutputStream OS;
    OS << std::numeric_limits<uint64_t>::max();
    EXPECT_EQ(OS.str(), "18446744073709551615");
  }
}

// A bare integer literal 0 is an int (identity match), not a null pointer
// constant, so it must print "0" rather than routing to the const void*
// overload as "0x0".
TEST(StringOutputStreamTest, LiteralZeroIsIntNotPointer) {
  StringOutputStream OS;
  OS << 0;
  EXPECT_EQ(OS.str(), "0");
}

TEST(StringOutputStreamTest, PointersAreHexByDefault) {
  StringOutputStream OS;
  const void *P = reinterpret_cast<const void *>(0xdeadbeefULL);
  OS << "p=" << P;
  EXPECT_EQ(OS.str(), "p=0xdeadbeef");
}

TEST(StringOutputStreamTest, NullPointerPrintsZero) {
  StringOutputStream OS;
  OS << static_cast<const void *>(nullptr);
  EXPECT_EQ(OS.str(), "0x0");
}

// Typed object pointers bind to the const void* overload (no dedicated
// template), so they too print as hex.
TEST(StringOutputStreamTest, TypedPointersAreHex) {
  int X = 0;
  StringOutputStream OS;
  OS << &X;
  EXPECT_EQ(OS.str().rfind("0x", 0), 0u); // starts with "0x"
  EXPECT_GT(OS.str().size(), 2u);
}

TEST(StringOutputStreamTest, HexBasics) {
  StringOutputStream OS;
  OS << hex(0) << ' ' << hex(255) << ' ' << hex(0xABCDu);
  EXPECT_EQ(OS.str(), "0x0 0xff 0xabcd");
}

// hex() prints the unsigned two's-complement bit pattern, and its width tracks
// the operand type rather than being widened to a pointer.
TEST(StringOutputStreamTest, HexNegativeIsTwosComplement) {
  {
    StringOutputStream OS;
    OS << hex(static_cast<int32_t>(-1));
    EXPECT_EQ(OS.str(), "0xffffffff");
  }
  {
    StringOutputStream OS;
    OS << hex(static_cast<int64_t>(-1));
    EXPECT_EQ(OS.str(), "0xffffffffffffffff");
  }
  {
    StringOutputStream OS;
    OS << hex(static_cast<uint8_t>(0x0f));
    EXPECT_EQ(OS.str(), "0xf"); // leading zeros dropped
  }
}

TEST(StringOutputStreamTest, ChainingReturnsSameStream) {
  StringOutputStream OS;
  StringOutputStream &Ref = (OS << "a" << 1 << hex(2));
  EXPECT_EQ(&Ref, &OS);
  EXPECT_EQ(OS.str(), "a10x2");
}

TEST(StringOutputStreamTest, RvalueStrMovesOut) {
  StringOutputStream OS;
  OS << "payload " << 123;
  std::string Moved = std::move(OS).str();
  EXPECT_EQ(Moved, "payload 123");
}

TEST(StringOutputStreamTest, RealisticMessage) {
  int Code = -22;
  const void *Addr = reinterpret_cast<const void *>(0x1000ULL);
  StringOutputStream OS;
  OS << "map failed at " << Addr << " (errno=" << Code
     << ", flags=" << hex(0x1Bu) << ")";
  EXPECT_EQ(OS.str(), "map failed at 0x1000 (errno=-22, flags=0x1b)");
}

TEST(StringOutputStreamTest, LeftJustifyPadsOnRight) {
  StringOutputStream OS;
  OS << '[' << ljust("ab", 5) << ']';
  EXPECT_EQ(OS.str(), "[ab   ]");
}

TEST(StringOutputStreamTest, RightJustifyPadsOnLeft) {
  StringOutputStream OS;
  OS << '[' << rjust("ab", 5) << ']';
  EXPECT_EQ(OS.str(), "[   ab]");
}

TEST(StringOutputStreamTest, JustifyCustomFill) {
  StringOutputStream OS;
  OS << ljust("ab", 5, '.') << rjust("cd", 5, '.');
  EXPECT_EQ(OS.str(), "ab......cd");
}

// A field at least as wide as the requested width is emitted in full, never
// truncated, and adds no padding.
TEST(StringOutputStreamTest, JustifyNoTruncationAtOrOverWidth) {
  {
    StringOutputStream OS;
    OS << '[' << ljust("exact", 5) << ']';
    EXPECT_EQ(OS.str(), "[exact]");
  }
  {
    StringOutputStream OS;
    OS << '[' << rjust("toolong", 3) << ']';
    EXPECT_EQ(OS.str(), "[toolong]");
  }
}

// Mirrors printHelp's column layout: a left-justified flag field followed by
// its description, so descriptions align regardless of flag length.
TEST(StringOutputStreamTest, JustifyAlignsColumns) {
  StringOutputStream OS;
  OS << ljust("--a", 10) << "first\n" << ljust("--bbbbb", 10) << "second\n";
  EXPECT_EQ(OS.str(), "--a       first\n--bbbbb   second\n");
}
