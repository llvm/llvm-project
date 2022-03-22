//===- llvm/unittest/Support/ConvertUTFTest.cpp - ConvertUTF tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ConvertUTF.h"
#include "llvm/ADT/ArrayRef.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace llvm;

TEST(ConvertUTFTest, ConvertUTF16LittleEndianToUTF8String) {
  // Src is the look of disapproval.
  alignas(UTF16) static const char Src[] = "\xff\xfe\xa0\x0c_\x00\xa0\x0c";
  ArrayRef<char> Ref(Src, sizeof(Src) - 1);
  std::string Result;
  bool Success = convertUTF16ToUTF8String(Ref, Result);
  EXPECT_TRUE(Success);
  std::string Expected("\xe0\xb2\xa0_\xe0\xb2\xa0");
  EXPECT_EQ(Expected, Result);
}

TEST(ConvertUTFTest, ConvertUTF16BigEndianToUTF8String) {
  // Src is the look of disapproval.
  alignas(UTF16) static const char Src[] = "\xfe\xff\x0c\xa0\x00_\x0c\xa0";
  ArrayRef<char> Ref(Src, sizeof(Src) - 1);
  std::string Result;
  bool Success = convertUTF16ToUTF8String(Ref, Result);
  EXPECT_TRUE(Success);
  std::string Expected("\xe0\xb2\xa0_\xe0\xb2\xa0");
  EXPECT_EQ(Expected, Result);
}

TEST(ConvertUTFTest, ConvertUTF8ToUTF16String) {
  // Src is the look of disapproval.
  static const char Src[] = "\xe0\xb2\xa0_\xe0\xb2\xa0";
  StringRef Ref(Src, sizeof(Src) - 1);
  SmallVector<UTF16, 5> Result;
  bool Success = convertUTF8ToUTF16String(Ref, Result);
  EXPECT_TRUE(Success);
  static const UTF16 Expected[] = {0x0CA0, 0x005f, 0x0CA0, 0};
  ASSERT_EQ(3u, Result.size());
  for (int I = 0, E = 3; I != E; ++I)
    EXPECT_EQ(Expected[I], Result[I]);
}

TEST(ConvertUTFTest, OddLengthInput) {
  std::string Result;
  bool Success = convertUTF16ToUTF8String(makeArrayRef("xxxxx", 5), Result);
  EXPECT_FALSE(Success);
}

TEST(ConvertUTFTest, Empty) {
  std::string Result;
  bool Success = convertUTF16ToUTF8String(llvm::ArrayRef<char>(None), Result);
  EXPECT_TRUE(Success);
  EXPECT_TRUE(Result.empty());
}

TEST(ConvertUTFTest, HasUTF16BOM) {
  bool HasBOM = hasUTF16ByteOrderMark(makeArrayRef("\xff\xfe", 2));
  EXPECT_TRUE(HasBOM);
  HasBOM = hasUTF16ByteOrderMark(makeArrayRef("\xfe\xff", 2));
  EXPECT_TRUE(HasBOM);
  HasBOM = hasUTF16ByteOrderMark(makeArrayRef("\xfe\xff ", 3));
  EXPECT_TRUE(HasBOM); // Don't care about odd lengths.
  HasBOM = hasUTF16ByteOrderMark(makeArrayRef("\xfe\xff\x00asdf", 6));
  EXPECT_TRUE(HasBOM);

  HasBOM = hasUTF16ByteOrderMark(None);
  EXPECT_FALSE(HasBOM);
  HasBOM = hasUTF16ByteOrderMark(makeArrayRef("\xfe", 1));
  EXPECT_FALSE(HasBOM);
}

TEST(ConvertUTFTest, UTF16WrappersForConvertUTF16ToUTF8String) {
  // Src is the look of disapproval.
  alignas(UTF16) static const char Src[] = "\xff\xfe\xa0\x0c_\x00\xa0\x0c";
  ArrayRef<UTF16> SrcRef = makeArrayRef((const UTF16 *)Src, 4);
  std::string Result;
  bool Success = convertUTF16ToUTF8String(SrcRef, Result);
  EXPECT_TRUE(Success);
  std::string Expected("\xe0\xb2\xa0_\xe0\xb2\xa0");
  EXPECT_EQ(Expected, Result);
}

TEST(ConvertUTFTest, ConvertUTF8toWide) {
  // Src is the look of disapproval.
  static const char Src[] = "\xe0\xb2\xa0_\xe0\xb2\xa0";
  std::wstring Result;
  bool Success = ConvertUTF8toWide((const char*)Src, Result);
  EXPECT_TRUE(Success);
  std::wstring Expected(L"\x0ca0_\x0ca0");
  EXPECT_EQ(Expected, Result);
  Result.clear();
  Success = ConvertUTF8toWide(StringRef(Src, 7), Result);
  EXPECT_TRUE(Success);
  EXPECT_EQ(Expected, Result);
}

TEST(ConvertUTFTest, convertWideToUTF8) {
  // Src is the look of disapproval.
  static const wchar_t Src[] = L"\x0ca0_\x0ca0";
  std::string Result;
  bool Success = convertWideToUTF8(Src, Result);
  EXPECT_TRUE(Success);
  std::string Expected("\xe0\xb2\xa0_\xe0\xb2\xa0");
  EXPECT_EQ(Expected, Result);
}

struct ConvertUTFResultContainer {
  ConversionResult ErrorCode;
  std::vector<unsigned> UnicodeScalars;

  ConvertUTFResultContainer(ConversionResult ErrorCode)
      : ErrorCode(ErrorCode) {}

  ConvertUTFResultContainer
  withScalars(unsigned US0 = 0x110000, unsigned US1 = 0x110000,
              unsigned US2 = 0x110000, unsigned US3 = 0x110000,
              unsigned US4 = 0x110000, unsigned US5 = 0x110000,
              unsigned US6 = 0x110000, unsigned US7 = 0x110000) {
    ConvertUTFResultContainer Result(*this);
    if (US0 != 0x110000)
      Result.UnicodeScalars.push_back(US0);
    if (US1 != 0x110000)
      Result.UnicodeScalars.push_back(US1);
    if (US2 != 0x110000)
      Result.UnicodeScalars.push_back(US2);
    if (US3 != 0x110000)
      Result.UnicodeScalars.push_back(US3);
    if (US4 != 0x110000)
      Result.UnicodeScalars.push_back(US4);
    if (US5 != 0x110000)
      Result.UnicodeScalars.push_back(US5);
    if (US6 != 0x110000)
      Result.UnicodeScalars.push_back(US6);
    if (US7 != 0x110000)
      Result.UnicodeScalars.push_back(US7);
    return Result;
  }
};

std::pair<ConversionResult, std::vector<unsigned>>
ConvertUTF8ToUnicodeScalarsLenient(StringRef S) {
  const UTF8 *SourceStart = reinterpret_cast<const UTF8 *>(S.data());

  const UTF8 *SourceNext = SourceStart;
  std::vector<UTF32> Decoded(S.size(), 0);
  UTF32 *TargetStart = Decoded.data();

  auto ErrorCode =
      ConvertUTF8toUTF32(&SourceNext, SourceStart + S.size(), &TargetStart,
                         Decoded.data() + Decoded.size(), lenientConversion);

  Decoded.resize(TargetStart - Decoded.data());

  return std::make_pair(ErrorCode, Decoded);
}

std::pair<ConversionResult, std::vector<unsigned>>
ConvertUTF8ToUnicodeScalarsPartialLenient(StringRef S) {
  const UTF8 *SourceStart = reinterpret_cast<const UTF8 *>(S.data());

  const UTF8 *SourceNext = SourceStart;
  std::vector<UTF32> Decoded(S.size(), 0);
  UTF32 *TargetStart = Decoded.data();

  auto ErrorCode = ConvertUTF8toUTF32Partial(
      &SourceNext, SourceStart + S.size(), &TargetStart,
      Decoded.data() + Decoded.size(), lenientConversion);

  Decoded.resize(TargetStart - Decoded.data());

  return std::make_pair(ErrorCode, Decoded);
}

::testing::AssertionResult
CheckConvertUTF8ToUnicodeScalars(ConvertUTFResultContainer Expected,
                                 StringRef S, bool Partial = false) {
  ConversionResult ErrorCode;
  std::vector<unsigned> Decoded;
  if (!Partial)
    std::tie(ErrorCode, Decoded) = ConvertUTF8ToUnicodeScalarsLenient(S);
  else
    std::tie(ErrorCode, Decoded) = ConvertUTF8ToUnicodeScalarsPartialLenient(S);

  if (Expected.ErrorCode != ErrorCode)
    return ::testing::AssertionFailure() << "Expected error code "
                                         << Expected.ErrorCode << ", actual "
                                         << ErrorCode;

  if (Expected.UnicodeScalars != Decoded)
    return ::testing::AssertionFailure()
           << "Expected lenient decoded result:\n"
           << ::testing::PrintToString(Expected.UnicodeScalars) << "\n"
           << "Actual result:\n" << ::testing::PrintToString(Decoded);

  return ::testing::AssertionSuccess();
}

TEST(ConvertUTFTest, UTF8ToUTF32Lenient) {

  //
  // 1-byte sequences
  //

  // U+0041 LATIN CAPITAL LETTER A
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x0041), "\x41"));

  //
  // 2-byte sequences
  //

  // U+0283 LATIN SMALL LETTER ESH
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x0283),
      "\xca\x83"));

  // U+03BA GREEK SMALL LETTER KAPPA
  // U+1F79 GREEK SMALL LETTER OMICRON WITH OXIA
  // U+03C3 GREEK SMALL LETTER SIGMA
  // U+03BC GREEK SMALL LETTER MU
  // U+03B5 GREEK SMALL LETTER EPSILON
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK)
          .withScalars(0x03ba, 0x1f79, 0x03c3, 0x03bc, 0x03b5),
      "\xce\xba\xe1\xbd\xb9\xcf\x83\xce\xbc\xce\xb5"));

  //
  // 3-byte sequences
  //

  // U+4F8B CJK UNIFIED IDEOGRAPH-4F8B
  // U+6587 CJK UNIFIED IDEOGRAPH-6587
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x4f8b, 0x6587),
      "\xe4\xbe\x8b\xe6\x96\x87"));

  // U+D55C HANGUL SYLLABLE HAN
  // U+AE00 HANGUL SYLLABLE GEUL
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xd55c, 0xae00),
      "\xed\x95\x9c\xea\xb8\x80"));

  // U+1112 HANGUL CHOSEONG HIEUH
  // U+1161 HANGUL JUNGSEONG A
  // U+11AB HANGUL JONGSEONG NIEUN
  // U+1100 HANGUL CHOSEONG KIYEOK
  // U+1173 HANGUL JUNGSEONG EU
  // U+11AF HANGUL JONGSEONG RIEUL
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK)
          .withScalars(0x1112, 0x1161, 0x11ab, 0x1100, 0x1173, 0x11af),
      "\xe1\x84\x92\xe1\x85\xa1\xe1\x86\xab\xe1\x84\x80\xe1\x85\xb3"
      "\xe1\x86\xaf"));

  //
  // 4-byte sequences
  //

  // U+E0100 VARIATION SELECTOR-17
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x000E0100),
      "\xf3\xa0\x84\x80"));

  //
  // First possible sequence of a certain length
  //

  // U+0000 NULL
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x0000),
      StringRef("\x00", 1)));

  // U+0080 PADDING CHARACTER
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x0080),
      "\xc2\x80"));

  // U+0800 SAMARITAN LETTER ALAF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x0800),
      "\xe0\xa0\x80"));

  // U+10000 LINEAR B SYLLABLE B008 A
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x10000),
      "\xf0\x90\x80\x80"));

  // U+200000 (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf8\x88\x80\x80\x80"));

  // U+4000000 (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfc\x84\x80\x80\x80\x80"));

  //
  // Last possible sequence of a certain length
  //

  // U+007F DELETE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x007f), "\x7f"));

  // U+07FF (unassigned)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x07ff),
      "\xdf\xbf"));

  // U+FFFF (noncharacter)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xffff),
      "\xef\xbf\xbf"));

  // U+1FFFFF (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf7\xbf\xbf\xbf"));

  // U+3FFFFFF (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfb\xbf\xbf\xbf\xbf"));

  // U+7FFFFFFF (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfd\xbf\xbf\xbf\xbf\xbf"));

  //
  // Other boundary conditions
  //

  // U+D7FF (unassigned)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xd7ff),
      "\xed\x9f\xbf"));

  // U+E000 (private use)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xe000),
      "\xee\x80\x80"));

  // U+FFFD REPLACEMENT CHARACTER
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfffd),
      "\xef\xbf\xbd"));

  // U+10FFFF (noncharacter)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x10ffff),
      "\xf4\x8f\xbf\xbf"));

  // U+110000 (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf4\x90\x80\x80"));

  //
  // Unexpected continuation bytes
  //

  // A sequence of unexpected continuation bytes that don't follow a first
  // byte, every byte is a maximal subpart.

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\x80\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xbf\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\x80\xbf\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\x80\xbf\x80\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\x80\xbf\x82\xbf\xaa"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xaa\xb0\xbb\xbf\xaa\xa0"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xaa\xb0\xbb\xbf\xaa\xa0\x8f"));

  // All continuation bytes (0x80--0xbf).
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"
      "\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"
      "\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"
      "\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"));

  //
  // Lonely start bytes
  //

  // Start bytes of 2-byte sequences (0xc0--0xdf).
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"
      "\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf"));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020),
      "\xc0\x20\xc1\x20\xc2\x20\xc3\x20\xc4\x20\xc5\x20\xc6\x20\xc7\x20"
      "\xc8\x20\xc9\x20\xca\x20\xcb\x20\xcc\x20\xcd\x20\xce\x20\xcf\x20"
      "\xd0\x20\xd1\x20\xd2\x20\xd3\x20\xd4\x20\xd5\x20\xd6\x20\xd7\x20"
      "\xd8\x20\xd9\x20\xda\x20\xdb\x20\xdc\x20\xdd\x20\xde\x20\xdf\x20"));

  // Start bytes of 3-byte sequences (0xe0--0xef).
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef"));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020),
      "\xe0\x20\xe1\x20\xe2\x20\xe3\x20\xe4\x20\xe5\x20\xe6\x20\xe7\x20"
      "\xe8\x20\xe9\x20\xea\x20\xeb\x20\xec\x20\xed\x20\xee\x20\xef\x20"));

  // Start bytes of 4-byte sequences (0xf0--0xf7).
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd,
                       0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7"));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020),
      "\xf0\x20\xf1\x20\xf2\x20\xf3\x20\xf4\x20\xf5\x20\xf6\x20\xf7\x20"));

  // Start bytes of 5-byte sequences (0xf8--0xfb).
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf8\xf9\xfa\xfb"));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020),
      "\xf8\x20\xf9\x20\xfa\x20\xfb\x20"));

  // Start bytes of 6-byte sequences (0xfc--0xfd).
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xfc\xfd"));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020),
      "\xfc\x20\xfd\x20"));

  //
  // Other bytes (0xc0--0xc1, 0xfe--0xff).
  //

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xc0"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xc1"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xfe"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xff"));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xc0\xc1\xfe\xff"));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfe\xfe\xff\xff"));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfe\x80\x80\x80\x80\x80"));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xff\x80\x80\x80\x80\x80"));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0x0020, 0xfffd, 0x0020,
                       0xfffd, 0x0020, 0xfffd, 0x0020),
      "\xc0\x20\xc1\x20\xfe\x20\xff\x20"));

  //
  // Sequences with one continuation byte missing
  //

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xc2"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xdf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xe0\xa0"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xe0\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xe1\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xec\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xed\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xed\x9f"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xee\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xef\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf0\x90\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf0\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf1\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf3\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf4\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf4\x8f\xbf"));

  // Overlong sequences with one trailing byte missing.
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xc0"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xc1"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xe0\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xe0\x9f"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf0\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf0\x8f\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf8\x80\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfc\x80\x80\x80\x80"));

  // Sequences that represent surrogates with one trailing byte missing.
  // High surrogates
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xed\xa0"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xed\xac"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xed\xaf"));
  // Low surrogates
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xed\xb0"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xed\xb4"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xed\xbf"));

  // Ill-formed 4-byte sequences.
  // 11110zzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+1100xx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf4\x90\x80"));
  // U+13FBxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf4\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf5\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf6\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf7\x80\x80"));
  // U+1FFBxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf7\xbf\xbf"));

  // Ill-formed 5-byte sequences.
  // 111110uu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+2000xx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf8\x88\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf8\xbf\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf9\x80\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfa\x80\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfb\x80\x80\x80"));
  // U+3FFFFxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfb\xbf\xbf\xbf"));

  // Ill-formed 6-byte sequences.
  // 1111110u 10uuuuuu 10uzzzzz 10zzzyyyy 10yyyyxx 10xxxxxx
  // U+40000xx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfc\x84\x80\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfc\xbf\xbf\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfd\x80\x80\x80\x80"));
  // U+7FFFFFxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfd\xbf\xbf\xbf\xbf"));

  //
  // Sequences with two continuation bytes missing
  //

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf0\x90"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf0\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf1\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf3\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf4\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd),
      "\xf4\x8f"));

  // Overlong sequences with two trailing byte missing.
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xe0"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf0\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf0\x8f"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf8\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfc\x80\x80\x80"));

  // Sequences that represent surrogates with two trailing bytes missing.
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xed"));

  // Ill-formed 4-byte sequences.
  // 11110zzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+110yxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf4\x90"));
  // U+13Fyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf4\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf5\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf6\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf7\x80"));
  // U+1FFyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf7\xbf"));

  // Ill-formed 5-byte sequences.
  // 111110uu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+200yxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf8\x88\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf8\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xf9\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xfa\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xfb\x80\x80"));
  // U+3FFFyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xfb\xbf\xbf"));

  // Ill-formed 6-byte sequences.
  // 1111110u 10uuuuuu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+4000yxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfc\x84\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfc\xbf\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfd\x80\x80\x80"));
  // U+7FFFFyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfd\xbf\xbf\xbf"));

  //
  // Sequences with three continuation bytes missing
  //

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf0"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf1"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf2"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf3"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf4"));

  // Broken overlong sequences.
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf0"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf8\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xfc\x80\x80"));

  // Ill-formed 4-byte sequences.
  // 11110zzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+14yyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf5"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf6"));
  // U+1Cyyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf7"));

  // Ill-formed 5-byte sequences.
  // 111110uu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+20yyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf8\x88"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf8\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xf9\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xfa\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xfb\x80"));
  // U+3FCyyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xfb\xbf"));

  // Ill-formed 6-byte sequences.
  // 1111110u 10uuuuuu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+400yyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xfc\x84\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xfc\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xfd\x80\x80"));
  // U+7FFCyyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xfd\xbf\xbf"));

  //
  // Sequences with four continuation bytes missing
  //

  // Ill-formed 5-byte sequences.
  // 111110uu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+uzyyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf8"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf9"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xfa"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xfb"));
  // U+3zyyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xfb"));

  // Broken overlong sequences.
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xf8"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xfc\x80"));

  // Ill-formed 6-byte sequences.
  // 1111110u 10uuuuuu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+uzzyyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xfc\x84"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xfc\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xfd\x80"));
  // U+7Fzzyyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xfd\xbf"));

  //
  // Sequences with five continuation bytes missing
  //

  // Ill-formed 6-byte sequences.
  // 1111110u 10uuuuuu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+uzzyyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xfc"));
  // U+uuzzyyxx (invalid)
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd), "\xfd"));

  //
  // Consecutive sequences with trailing bytes missing
  //

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, /**/ 0xfffd, 0xfffd, /**/ 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, /**/ 0xfffd, /**/ 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xc0" "\xe0\x80" "\xf0\x80\x80"
      "\xf8\x80\x80\x80"
      "\xfc\x80\x80\x80\x80"
      "\xdf" "\xef\xbf" "\xf7\xbf\xbf"
      "\xfb\xbf\xbf\xbf"
      "\xfd\xbf\xbf\xbf\xbf"));

  //
  // Overlong UTF-8 sequences
  //

  // U+002F SOLIDUS
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x002f), "\x2f"));

  // Overlong sequences of the above.
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xc0\xaf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xe0\x80\xaf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf0\x80\x80\xaf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf8\x80\x80\x80\xaf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfc\x80\x80\x80\x80\xaf"));

  // U+0000 NULL
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x0000),
      StringRef("\x00", 1)));

  // Overlong sequences of the above.
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xc0\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xe0\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf0\x80\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf8\x80\x80\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfc\x80\x80\x80\x80\x80"));

  // Other overlong sequences.
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xc0\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xc1\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal).withScalars(0xfffd, 0xfffd),
      "\xc1\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xe0\x9f\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xed\xa0\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xed\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf0\x8f\x80\x80"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf0\x8f\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xf8\x87\xbf\xbf\xbf"));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xfc\x83\xbf\xbf\xbf\xbf"));

  //
  // Isolated surrogates
  //

  // Unicode 6.3.0:
  //
  //    D71.  High-surrogate code point: A Unicode code point in the range
  //    U+D800 to U+DBFF.
  //
  //    D73.  Low-surrogate code point: A Unicode code point in the range
  //    U+DC00 to U+DFFF.

  // Note: U+E0100 is <DB40 DD00> in UTF16.

  // High surrogates

  // U+D800
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xed\xa0\x80"));

  // U+DB40
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xed\xac\xa0"));

  // U+DBFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xed\xaf\xbf"));

  // Low surrogates

  // U+DC00
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xed\xb0\x80"));

  // U+DD00
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xed\xb4\x80"));

  // U+DFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd),
      "\xed\xbf\xbf"));

  // Surrogate pairs

  // U+D800 U+DC00
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xed\xa0\x80\xed\xb0\x80"));

  // U+D800 U+DD00
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xed\xa0\x80\xed\xb4\x80"));

  // U+D800 U+DFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xed\xa0\x80\xed\xbf\xbf"));

  // U+DB40 U+DC00
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xed\xac\xa0\xed\xb0\x80"));

  // U+DB40 U+DD00
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xed\xac\xa0\xed\xb4\x80"));

  // U+DB40 U+DFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xed\xac\xa0\xed\xbf\xbf"));

  // U+DBFF U+DC00
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xed\xaf\xbf\xed\xb0\x80"));

  // U+DBFF U+DD00
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xed\xaf\xbf\xed\xb4\x80"));

  // U+DBFF U+DFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceIllegal)
          .withScalars(0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      "\xed\xaf\xbf\xed\xbf\xbf"));

  //
  // Noncharacters
  //

  // Unicode 6.3.0:
  //
  //    D14.  Noncharacter: A code point that is permanently reserved for
  //    internal use and that should never be interchanged. Noncharacters
  //    consist of the values U+nFFFE and U+nFFFF (where n is from 0 to 1016)
  //    and the values U+FDD0..U+FDEF.

  // U+FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfffe),
      "\xef\xbf\xbe"));

  // U+FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xffff),
      "\xef\xbf\xbf"));

  // U+1FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x1fffe),
      "\xf0\x9f\xbf\xbe"));

  // U+1FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x1ffff),
      "\xf0\x9f\xbf\xbf"));

  // U+2FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x2fffe),
      "\xf0\xaf\xbf\xbe"));

  // U+2FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x2ffff),
      "\xf0\xaf\xbf\xbf"));

  // U+3FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x3fffe),
      "\xf0\xbf\xbf\xbe"));

  // U+3FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x3ffff),
      "\xf0\xbf\xbf\xbf"));

  // U+4FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x4fffe),
      "\xf1\x8f\xbf\xbe"));

  // U+4FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x4ffff),
      "\xf1\x8f\xbf\xbf"));

  // U+5FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x5fffe),
      "\xf1\x9f\xbf\xbe"));

  // U+5FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x5ffff),
      "\xf1\x9f\xbf\xbf"));

  // U+6FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x6fffe),
      "\xf1\xaf\xbf\xbe"));

  // U+6FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x6ffff),
      "\xf1\xaf\xbf\xbf"));

  // U+7FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x7fffe),
      "\xf1\xbf\xbf\xbe"));

  // U+7FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x7ffff),
      "\xf1\xbf\xbf\xbf"));

  // U+8FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x8fffe),
      "\xf2\x8f\xbf\xbe"));

  // U+8FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x8ffff),
      "\xf2\x8f\xbf\xbf"));

  // U+9FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x9fffe),
      "\xf2\x9f\xbf\xbe"));

  // U+9FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x9ffff),
      "\xf2\x9f\xbf\xbf"));

  // U+AFFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xafffe),
      "\xf2\xaf\xbf\xbe"));

  // U+AFFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xaffff),
      "\xf2\xaf\xbf\xbf"));

  // U+BFFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xbfffe),
      "\xf2\xbf\xbf\xbe"));

  // U+BFFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xbffff),
      "\xf2\xbf\xbf\xbf"));

  // U+CFFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xcfffe),
      "\xf3\x8f\xbf\xbe"));

  // U+CFFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xcfffF),
      "\xf3\x8f\xbf\xbf"));

  // U+DFFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xdfffe),
      "\xf3\x9f\xbf\xbe"));

  // U+DFFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xdffff),
      "\xf3\x9f\xbf\xbf"));

  // U+EFFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xefffe),
      "\xf3\xaf\xbf\xbe"));

  // U+EFFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xeffff),
      "\xf3\xaf\xbf\xbf"));

  // U+FFFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xffffe),
      "\xf3\xbf\xbf\xbe"));

  // U+FFFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfffff),
      "\xf3\xbf\xbf\xbf"));

  // U+10FFFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x10fffe),
      "\xf4\x8f\xbf\xbe"));

  // U+10FFFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x10ffff),
      "\xf4\x8f\xbf\xbf"));

  // U+FDD0
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdd0),
      "\xef\xb7\x90"));

  // U+FDD1
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdd1),
      "\xef\xb7\x91"));

  // U+FDD2
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdd2),
      "\xef\xb7\x92"));

  // U+FDD3
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdd3),
      "\xef\xb7\x93"));

  // U+FDD4
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdd4),
      "\xef\xb7\x94"));

  // U+FDD5
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdd5),
      "\xef\xb7\x95"));

  // U+FDD6
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdd6),
      "\xef\xb7\x96"));

  // U+FDD7
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdd7),
      "\xef\xb7\x97"));

  // U+FDD8
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdd8),
      "\xef\xb7\x98"));

  // U+FDD9
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdd9),
      "\xef\xb7\x99"));

  // U+FDDA
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdda),
      "\xef\xb7\x9a"));

  // U+FDDB
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfddb),
      "\xef\xb7\x9b"));

  // U+FDDC
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfddc),
      "\xef\xb7\x9c"));

  // U+FDDD
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfddd),
      "\xef\xb7\x9d"));

  // U+FDDE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdde),
      "\xef\xb7\x9e"));

  // U+FDDF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfddf),
      "\xef\xb7\x9f"));

  // U+FDE0
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfde0),
      "\xef\xb7\xa0"));

  // U+FDE1
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfde1),
      "\xef\xb7\xa1"));

  // U+FDE2
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfde2),
      "\xef\xb7\xa2"));

  // U+FDE3
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfde3),
      "\xef\xb7\xa3"));

  // U+FDE4
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfde4),
      "\xef\xb7\xa4"));

  // U+FDE5
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfde5),
      "\xef\xb7\xa5"));

  // U+FDE6
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfde6),
      "\xef\xb7\xa6"));

  // U+FDE7
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfde7),
      "\xef\xb7\xa7"));

  // U+FDE8
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfde8),
      "\xef\xb7\xa8"));

  // U+FDE9
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfde9),
      "\xef\xb7\xa9"));

  // U+FDEA
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdea),
      "\xef\xb7\xaa"));

  // U+FDEB
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdeb),
      "\xef\xb7\xab"));

  // U+FDEC
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdec),
      "\xef\xb7\xac"));

  // U+FDED
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfded),
      "\xef\xb7\xad"));

  // U+FDEE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdee),
      "\xef\xb7\xae"));

  // U+FDEF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdef),
      "\xef\xb7\xaf"));

  // U+FDF0
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdf0),
      "\xef\xb7\xb0"));

  // U+FDF1
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdf1),
      "\xef\xb7\xb1"));

  // U+FDF2
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdf2),
      "\xef\xb7\xb2"));

  // U+FDF3
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdf3),
      "\xef\xb7\xb3"));

  // U+FDF4
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdf4),
      "\xef\xb7\xb4"));

  // U+FDF5
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdf5),
      "\xef\xb7\xb5"));

  // U+FDF6
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdf6),
      "\xef\xb7\xb6"));

  // U+FDF7
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdf7),
      "\xef\xb7\xb7"));

  // U+FDF8
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdf8),
      "\xef\xb7\xb8"));

  // U+FDF9
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdf9),
      "\xef\xb7\xb9"));

  // U+FDFA
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdfa),
      "\xef\xb7\xba"));

  // U+FDFB
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdfb),
      "\xef\xb7\xbb"));

  // U+FDFC
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdfc),
      "\xef\xb7\xbc"));

  // U+FDFD
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdfd),
      "\xef\xb7\xbd"));

  // U+FDFE
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdfe),
      "\xef\xb7\xbe"));

  // U+FDFF
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0xfdff),
      "\xef\xb7\xbf"));
}

TEST(ConvertUTFTest, UTF8ToUTF32PartialLenient) {
  // U+0041 LATIN CAPITAL LETTER A
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(conversionOK).withScalars(0x0041),
      "\x41", true));

  //
  // Sequences with one continuation byte missing
  //

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xc2", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xdf", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xe0\xa0", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xe0\xbf", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xe1\x80", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xec\xbf", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xed\x80", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xed\x9f", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xee\x80", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xef\xbf", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xf0\x90\x80", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xf0\xbf\xbf", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xf1\x80\x80", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xf3\xbf\xbf", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xf4\x80\x80", true));
  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted),
      "\xf4\x8f\xbf", true));

  EXPECT_TRUE(CheckConvertUTF8ToUnicodeScalars(
      ConvertUTFResultContainer(sourceExhausted).withScalars(0x0041),
      "\x41\xc2", true));
}

