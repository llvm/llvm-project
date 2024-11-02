//===- FormatChkTest.cpp - Unit tests for checked string formatting -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Format.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;

namespace {

constexpr auto ST_Unknown = llvm::PrintfStyleFormatReader::ST_Unknown;
constexpr auto ST_WideChar = llvm::PrintfStyleFormatReader::ST_WideChar;
constexpr auto ST_Int = llvm::PrintfStyleFormatReader::ST_Int;
constexpr auto ST_Long = llvm::PrintfStyleFormatReader::ST_Long;
constexpr auto ST_LongLong = llvm::PrintfStyleFormatReader::ST_LongLong;
constexpr auto ST_IntMax = llvm::PrintfStyleFormatReader::ST_IntMax;
constexpr auto ST_Size = llvm::PrintfStyleFormatReader::ST_Size;
constexpr auto ST_Ptrdiff = llvm::PrintfStyleFormatReader::ST_Ptrdiff;
constexpr auto ST_Double = llvm::PrintfStyleFormatReader::ST_Double;
constexpr auto ST_LongDouble = llvm::PrintfStyleFormatReader::ST_LongDouble;
constexpr auto ST_CString = llvm::PrintfStyleFormatReader::ST_CString;
constexpr auto ST_WideCString = llvm::PrintfStyleFormatReader::ST_WideCString;
constexpr auto ST_VoidPointer = llvm::PrintfStyleFormatReader::ST_VoidPointer;
constexpr auto ST_Count_Char = llvm::PrintfStyleFormatReader::ST_Count_Char;
constexpr auto ST_Count_Short = llvm::PrintfStyleFormatReader::ST_Count_Short;
constexpr auto ST_Count_Int = llvm::PrintfStyleFormatReader::ST_Count_Int;
constexpr auto ST_Count_Long = llvm::PrintfStyleFormatReader::ST_Count_Long;
constexpr auto ST_Count_LongLong =
    llvm::PrintfStyleFormatReader::ST_Count_LongLong;
constexpr auto ST_Count_IntMax = llvm::PrintfStyleFormatReader::ST_Count_IntMax;
constexpr auto ST_Count_Size = llvm::PrintfStyleFormatReader::ST_Count_Size;
constexpr auto ST_Count_Ptrdiff =
    llvm::PrintfStyleFormatReader::ST_Count_Ptrdiff;

using STVec = std::vector<PrintfStyleFormatReader::SpecifierType>;

STVec ParseFormatString(const char *Fmt) {
  STVec Result;
  PrintfStyleFormatReader Reader(Fmt);
  while (auto Spec = Reader.nextSpecifier()) {
    Result.push_back(Spec);
  }
  return Result;
}

#define EXPECT_FMT_EQ(FMT, ...)                                                \
  EXPECT_EQ(ParseFormatString(FMT), STVec({__VA_ARGS__}))

} // namespace

TEST(FormatReader, EmptyFormatString) {
  EXPECT_EQ(ParseFormatString(""),
            std::vector<PrintfStyleFormatReader::SpecifierType>());
}

TEST(FormatReader, PercentEscape) {
  EXPECT_EQ(ParseFormatString("%%"),
            std::vector<PrintfStyleFormatReader::SpecifierType>());
}

TEST(FormatReader, PercentAtEnd) { EXPECT_FMT_EQ("%", ST_Unknown); }

TEST(FormatReader, PercentWithWidth) { EXPECT_FMT_EQ("%ll%", ST_Unknown); }

TEST(FormatReader, OneFormat) {
  EXPECT_FMT_EQ("%i xx", ST_Int);
  EXPECT_FMT_EQ("yy %i", ST_Int);
  EXPECT_FMT_EQ("yy %i xx", ST_Int);
}

TEST(FormatReader, TwoFormats) {
  EXPECT_FMT_EQ("%i yy %f xx", ST_Int, ST_Double);
  EXPECT_FMT_EQ("zz %i yy %f", ST_Int, ST_Double);
  EXPECT_FMT_EQ("zz %i yy %f xx", ST_Int, ST_Double);
}

TEST(FormatReader, PoundFlagValid) {
  EXPECT_FMT_EQ("%#x", ST_Int);
  EXPECT_FMT_EQ("%#X", ST_Int);
  EXPECT_FMT_EQ("%#a", ST_Double);
  EXPECT_FMT_EQ("%#A", ST_Double);
  EXPECT_FMT_EQ("%#e", ST_Double);
  EXPECT_FMT_EQ("%#E", ST_Double);
  EXPECT_FMT_EQ("%#f", ST_Double);
  EXPECT_FMT_EQ("%#F", ST_Double);
  EXPECT_FMT_EQ("%#g", ST_Double);
  EXPECT_FMT_EQ("%#G", ST_Double);

  EXPECT_FMT_EQ("%#p", ST_Unknown);
  EXPECT_FMT_EQ("%#i", ST_Unknown);
  EXPECT_FMT_EQ("%#c", ST_Unknown);
  EXPECT_FMT_EQ("%#s", ST_Unknown);
  EXPECT_FMT_EQ("%#d", ST_Unknown);
  EXPECT_FMT_EQ("%#u", ST_Unknown);
  EXPECT_FMT_EQ("%#o", ST_Unknown);
  EXPECT_FMT_EQ("%#n", ST_Unknown);
}

TEST(FormatReader, ZeroFlagValid) {
  EXPECT_FMT_EQ("%0x", ST_Int);
  EXPECT_FMT_EQ("%0X", ST_Int);
  EXPECT_FMT_EQ("%0i", ST_Int);
  EXPECT_FMT_EQ("%0d", ST_Int);
  EXPECT_FMT_EQ("%0u", ST_Int);
  EXPECT_FMT_EQ("%0o", ST_Int);
  EXPECT_FMT_EQ("%0a", ST_Double);
  EXPECT_FMT_EQ("%0A", ST_Double);
  EXPECT_FMT_EQ("%0e", ST_Double);
  EXPECT_FMT_EQ("%0E", ST_Double);
  EXPECT_FMT_EQ("%0f", ST_Double);
  EXPECT_FMT_EQ("%0F", ST_Double);
  EXPECT_FMT_EQ("%0g", ST_Double);
  EXPECT_FMT_EQ("%0G", ST_Double);

  EXPECT_FMT_EQ("%0p", ST_Unknown);
  EXPECT_FMT_EQ("%0n", ST_Unknown);
  EXPECT_FMT_EQ("%0c", ST_Unknown);
  EXPECT_FMT_EQ("%0s", ST_Unknown);
}

TEST(FormatReader, PrecisionValid) {
  EXPECT_FMT_EQ("%.1x", ST_Int);
  EXPECT_FMT_EQ("%.1X", ST_Int);
  EXPECT_FMT_EQ("%.1i", ST_Int);
  EXPECT_FMT_EQ("%.1d", ST_Int);
  EXPECT_FMT_EQ("%.1u", ST_Int);
  EXPECT_FMT_EQ("%.1o", ST_Int);
  EXPECT_FMT_EQ("%.1a", ST_Double);
  EXPECT_FMT_EQ("%.1A", ST_Double);
  EXPECT_FMT_EQ("%.1e", ST_Double);
  EXPECT_FMT_EQ("%.1E", ST_Double);
  EXPECT_FMT_EQ("%.1f", ST_Double);
  EXPECT_FMT_EQ("%.1F", ST_Double);
  EXPECT_FMT_EQ("%.1g", ST_Double);
  EXPECT_FMT_EQ("%.1G", ST_Double);
  EXPECT_FMT_EQ("%.1s", ST_CString);

  EXPECT_FMT_EQ("%.1p", ST_Unknown);
  EXPECT_FMT_EQ("%.1n", ST_Unknown);
  EXPECT_FMT_EQ("%.1c", ST_Unknown);
}

TEST(FormatReader, LongWidth) {
  EXPECT_FMT_EQ("%1li", ST_Long);
  EXPECT_FMT_EQ("%11li", ST_Long);
  EXPECT_FMT_EQ("%1111li", ST_Long);
  EXPECT_FMT_EQ("%10li", ST_Long);
  EXPECT_FMT_EQ("%*li", ST_Int, ST_Long);
  EXPECT_FMT_EQ("%*l!", ST_Unknown);
}

TEST(FormatReader, LongPrecision) {
  EXPECT_FMT_EQ("%.1li", ST_Long);
  EXPECT_FMT_EQ("%.11li", ST_Long);
  EXPECT_FMT_EQ("%.1111li", ST_Long);
  EXPECT_FMT_EQ("%.10li", ST_Long);
  EXPECT_FMT_EQ("%.*li", ST_Int, ST_Long);
  EXPECT_FMT_EQ("%.*l!", ST_Unknown);

  EXPECT_FMT_EQ("%1.1li", ST_Long);
  EXPECT_FMT_EQ("%11.11li", ST_Long);
  EXPECT_FMT_EQ("%111.1111li", ST_Long);
  EXPECT_FMT_EQ("%110.10li", ST_Long);
  EXPECT_FMT_EQ("%1.*li", ST_Int, ST_Long);
  EXPECT_FMT_EQ("%1.*l!", ST_Unknown);

  EXPECT_FMT_EQ("%*.*li", ST_Int, ST_Int, ST_Long);
  EXPECT_FMT_EQ("%*.*l!", ST_Unknown);
}

TEST(FormatReader, IntSpecifiers) {
  EXPECT_FMT_EQ("%hhi", ST_Int);
  EXPECT_FMT_EQ("%hhd", ST_Int);
  EXPECT_FMT_EQ("%hi", ST_Int);
  EXPECT_FMT_EQ("%hd", ST_Int);
  EXPECT_FMT_EQ("%i", ST_Int);
  EXPECT_FMT_EQ("%d", ST_Int);
  EXPECT_FMT_EQ("%li", ST_Long);
  EXPECT_FMT_EQ("%ld", ST_Long);
  EXPECT_FMT_EQ("%lli", ST_LongLong);
  EXPECT_FMT_EQ("%lld", ST_LongLong);
  EXPECT_FMT_EQ("%ji", ST_IntMax);
  EXPECT_FMT_EQ("%jd", ST_IntMax);
  EXPECT_FMT_EQ("%zi", ST_Size);
  EXPECT_FMT_EQ("%zd", ST_Size);
  EXPECT_FMT_EQ("%ti", ST_Ptrdiff);
  EXPECT_FMT_EQ("%td", ST_Ptrdiff);

  EXPECT_FMT_EQ("%Li", ST_Unknown);
  EXPECT_FMT_EQ("%Ld", ST_Unknown);
}

TEST(FormatReader, UIntSpecifiers) {
  EXPECT_FMT_EQ("%hhu", ST_Int);
  EXPECT_FMT_EQ("%hho", ST_Int);
  EXPECT_FMT_EQ("%hhx", ST_Int);
  EXPECT_FMT_EQ("%hhX", ST_Int);
  EXPECT_FMT_EQ("%hu", ST_Int);
  EXPECT_FMT_EQ("%ho", ST_Int);
  EXPECT_FMT_EQ("%hx", ST_Int);
  EXPECT_FMT_EQ("%hX", ST_Int);
  EXPECT_FMT_EQ("%u", ST_Int);
  EXPECT_FMT_EQ("%o", ST_Int);
  EXPECT_FMT_EQ("%x", ST_Int);
  EXPECT_FMT_EQ("%X", ST_Int);
  EXPECT_FMT_EQ("%lu", ST_Long);
  EXPECT_FMT_EQ("%lo", ST_Long);
  EXPECT_FMT_EQ("%lx", ST_Long);
  EXPECT_FMT_EQ("%lX", ST_Long);
  EXPECT_FMT_EQ("%llu", ST_LongLong);
  EXPECT_FMT_EQ("%llo", ST_LongLong);
  EXPECT_FMT_EQ("%llx", ST_LongLong);
  EXPECT_FMT_EQ("%llX", ST_LongLong);
  EXPECT_FMT_EQ("%ju", ST_IntMax);
  EXPECT_FMT_EQ("%jo", ST_IntMax);
  EXPECT_FMT_EQ("%jx", ST_IntMax);
  EXPECT_FMT_EQ("%jX", ST_IntMax);
  EXPECT_FMT_EQ("%zu", ST_Size);
  EXPECT_FMT_EQ("%zo", ST_Size);
  EXPECT_FMT_EQ("%zx", ST_Size);
  EXPECT_FMT_EQ("%zX", ST_Size);
  EXPECT_FMT_EQ("%tu", ST_Ptrdiff);
  EXPECT_FMT_EQ("%to", ST_Ptrdiff);
  EXPECT_FMT_EQ("%tx", ST_Ptrdiff);
  EXPECT_FMT_EQ("%tX", ST_Ptrdiff);

  EXPECT_FMT_EQ("%Lu", ST_Unknown);
  EXPECT_FMT_EQ("%Lo", ST_Unknown);
  EXPECT_FMT_EQ("%Lx", ST_Unknown);
  EXPECT_FMT_EQ("%LX", ST_Unknown);
}

TEST(FormatReader, FloatSpecifiers) {
  EXPECT_FMT_EQ("%a", ST_Double);
  EXPECT_FMT_EQ("%e", ST_Double);
  EXPECT_FMT_EQ("%f", ST_Double);
  EXPECT_FMT_EQ("%g", ST_Double);
  EXPECT_FMT_EQ("%la", ST_Double);
  EXPECT_FMT_EQ("%le", ST_Double);
  EXPECT_FMT_EQ("%lf", ST_Double);
  EXPECT_FMT_EQ("%lg", ST_Double);

  EXPECT_FMT_EQ("%La", ST_LongDouble);
  EXPECT_FMT_EQ("%Le", ST_LongDouble);
  EXPECT_FMT_EQ("%Lf", ST_LongDouble);
  EXPECT_FMT_EQ("%Lg", ST_LongDouble);

  EXPECT_FMT_EQ("%ha", ST_Unknown);
  EXPECT_FMT_EQ("%he", ST_Unknown);
  EXPECT_FMT_EQ("%hf", ST_Unknown);
  EXPECT_FMT_EQ("%hg", ST_Unknown);
  EXPECT_FMT_EQ("%hha", ST_Unknown);
  EXPECT_FMT_EQ("%hhe", ST_Unknown);
  EXPECT_FMT_EQ("%hhf", ST_Unknown);
  EXPECT_FMT_EQ("%hhg", ST_Unknown);
  EXPECT_FMT_EQ("%lla", ST_Unknown);
  EXPECT_FMT_EQ("%lle", ST_Unknown);
  EXPECT_FMT_EQ("%llf", ST_Unknown);
  EXPECT_FMT_EQ("%llg", ST_Unknown);
}

TEST(FormatReader, CharSpecifiers) {
  EXPECT_FMT_EQ("%hhc", ST_Unknown);
  EXPECT_FMT_EQ("%hc", ST_Unknown);
  EXPECT_FMT_EQ("%c", ST_Int);
  EXPECT_FMT_EQ("%lc", ST_WideChar);
  EXPECT_FMT_EQ("%llc", ST_Unknown);
  EXPECT_FMT_EQ("%jc", ST_Unknown);
  EXPECT_FMT_EQ("%zc", ST_Unknown);
  EXPECT_FMT_EQ("%tc", ST_Unknown);
  EXPECT_FMT_EQ("%Lc", ST_Unknown);
}

TEST(FormatReader, StringSpecifiers) {
  EXPECT_FMT_EQ("%hhs", ST_Unknown);
  EXPECT_FMT_EQ("%hs", ST_Unknown);
  EXPECT_FMT_EQ("%s", ST_CString);
  EXPECT_FMT_EQ("%ls", ST_WideCString);
  EXPECT_FMT_EQ("%lls", ST_Unknown);
  EXPECT_FMT_EQ("%js", ST_Unknown);
  EXPECT_FMT_EQ("%zs", ST_Unknown);
  EXPECT_FMT_EQ("%ts", ST_Unknown);
  EXPECT_FMT_EQ("%Ls", ST_Unknown);
}

TEST(FormatReader, VoidPointerSpecifiers) {
  EXPECT_FMT_EQ("%hhp", ST_Unknown);
  EXPECT_FMT_EQ("%hp", ST_Unknown);
  EXPECT_FMT_EQ("%p", ST_VoidPointer);
  EXPECT_FMT_EQ("%lp", ST_Unknown);
  EXPECT_FMT_EQ("%llp", ST_Unknown);
  EXPECT_FMT_EQ("%jp", ST_Unknown);
  EXPECT_FMT_EQ("%zp", ST_Unknown);
  EXPECT_FMT_EQ("%tp", ST_Unknown);
  EXPECT_FMT_EQ("%Lp", ST_Unknown);
}

TEST(FormatReader, CountSpecifiers) {
  EXPECT_FMT_EQ("%hhn", ST_Count_Char);
  EXPECT_FMT_EQ("%hn", ST_Count_Short);
  EXPECT_FMT_EQ("%n", ST_Count_Int);
  EXPECT_FMT_EQ("%ln", ST_Count_Long);
  EXPECT_FMT_EQ("%lln", ST_Count_LongLong);
  EXPECT_FMT_EQ("%jn", ST_Count_IntMax);
  EXPECT_FMT_EQ("%zn", ST_Count_Size);
  EXPECT_FMT_EQ("%tn", ST_Count_Ptrdiff);
  EXPECT_FMT_EQ("%Ln", ST_Unknown);
}
