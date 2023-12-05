//===- unittests/Support/CharSetTest.cpp - Charset conversion tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CharSet.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

// String "Hello World!"
static const char HelloA[] =
    "\x48\x65\x6C\x6C\x6F\x20\x57\x6F\x72\x6C\x64\x21\x0a";
static const char HelloE[] =
    "\xC8\x85\x93\x93\x96\x40\xE6\x96\x99\x93\x84\x5A\x15";

// String "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
static const char ABCStrA[] =
    "\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4A\x4B\x4C\x4D\x4E\x4F\x50\x51\x52"
    "\x53\x54\x55\x56\x57\x58\x59\x5A\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6A"
    "\x6B\x6C\x6D\x6E\x6F\x70\x71\x72\x73\x74\x75\x76\x77\x78\x79\x7A";
static const char ABCStrE[] =
    "\xC1\xC2\xC3\xC4\xC5\xC6\xC7\xC8\xC9\xD1\xD2\xD3\xD4\xD5\xD6\xD7\xD8\xD9"
    "\xE2\xE3\xE4\xE5\xE6\xE7\xE8\xE9\x81\x82\x83\x84\x85\x86\x87\x88\x89\x91"
    "\x92\x93\x94\x95\x96\x97\x98\x99\xA2\xA3\xA4\xA5\xA6\xA7\xA8\xA9";

// String "¡¢£AÄÅÆEÈÉÊaàáâãäeèéêë"
static const char AccentUTF[] =
    "\xc2\xa1\xc2\xa2\xc2\xa3\x41\xc3\x84\xc3\x85\xc3\x86\x45\xc3\x88\xc3\x89"
    "\xc3\x8a\x61\xc3\xa0\xc3\xa1\xc3\xa2\xc3\xa3\xc3\xa4\x65\xc3\xa8\xc3\xa9"
    "\xc3\xaa\xc3\xab";
static const char AccentE[] = "\xaa\x4a\xb1\xc1\x63\x67\x9e\xc5\x74\x71\x72"
                              "\x81\x44\x45\x42\x46\x43\x85\x54\x51\x52\x53";

// String with Cyrillic character ya.
static const char CyrillicUTF[] = "\xd0\xaf";

// String "Earth地球".
// ISO-2022-JP: Sequence ESC $ B (\x1B\x24\x42) switches to JIS X 0208-1983, and
// sequence ESC ( B (\x1B\x28\x42) switches back to ASCII.
// IBM-939: Byte 0x0E shifts from single byte to double byte, and 0x0F shifts
// back.
static const char EarthUTF[] = "\x45\x61\x72\x74\x68\xe5\x9c\xb0\xe7\x90\x83";
// Identical to above, except the final character (球) has its last byte taken
// away from it.
static const char EarthISO2022[] =
    "\x45\x61\x72\x74\x68\x1B\x24\x42\x43\x4F\x35\x65\x1B\x28\x42";
static const char EarthIBM939[] =
    "\xc5\x81\x99\xa3\x88\x0e\x45\xc2\x48\xdb\x0f";

TEST(CharSet, FromUTF8) {
  // Hello string.
  StringRef Src(HelloA);
  SmallString<64> Dst;

  CharSetConverter Conv = CharSetConverter::create(text_encoding::id::UTF8,
                                                   text_encoding::id::IBM1047);
  std::error_code EC = Conv.convert(Src, Dst, true);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(HelloE, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // ABC string.
  Src = ABCStrA;
  EC = Conv.convert(Src, Dst, true);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(ABCStrE, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // Accent string.
  Src = AccentUTF;
  EC = Conv.convert(Src, Dst, true);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(AccentE, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // Cyrillic string. Results in error because not representable in 1047.
  Src = CyrillicUTF;
  EC = Conv.convert(Src, Dst, true);
  EXPECT_EQ(EC, std::errc::illegal_byte_sequence);
}

TEST(CharSet, ToUTF8) {
  // Hello string.
  StringRef Src(HelloE);
  SmallString<64> Dst;

  CharSetConverter Conv = CharSetConverter::create(text_encoding::id::IBM1047,
                                                   text_encoding::id::UTF8);
  std::error_code EC = Conv.convert(Src, Dst, true);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(HelloA, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // ABC string.
  Src = ABCStrE;
  EC = Conv.convert(Src, Dst, true);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(ABCStrA, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // Accent string.
  Src = AccentE;
  EC = Conv.convert(Src, Dst, true);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(AccentUTF, static_cast<std::string>(Dst).c_str());
}

TEST(CharSet, RoundTrip) {
  ErrorOr<CharSetConverter> ConvToUTF16 =
      CharSetConverter::create("IBM-1047", "UTF-16");
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvToUTF16) {
    ASSERT_EQ(ConvToUTF16.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }
  ErrorOr<CharSetConverter> ConvToUTF32 =
      CharSetConverter::create("UTF-16", "UTF-32");
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvToUTF32) {
    ASSERT_EQ(ConvToUTF32.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }
  ErrorOr<CharSetConverter> ConvToEBCDIC =
      CharSetConverter::create("UTF-32", "IBM-1047");
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvToEBCDIC) {
    ASSERT_EQ(ConvToEBCDIC.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }

  // Setup source string.
  char SrcStr[256];
  for (size_t I = 0; I < 256; ++I)
    SrcStr[I] = (I + 1) % 256;

  SmallString<99> Dst1Str, Dst2Str, Dst3Str;

  std::error_code EC = ConvToUTF16->convert(StringRef(SrcStr), Dst1Str, true);
  EXPECT_TRUE(!EC);
  EC = ConvToUTF32->convert(Dst1Str, Dst2Str, true);
  EXPECT_TRUE(!EC);
  EC = ConvToEBCDIC->convert(Dst2Str, Dst3Str, true);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(SrcStr, static_cast<std::string>(Dst3Str).c_str());
}

TEST(CharSet, ShiftState2022) {
  // Earth string.
  StringRef Src(EarthUTF);
  SmallString<64> Dst;

  ErrorOr<CharSetConverter> ConvTo2022 =
      CharSetConverter::create("UTF-8", "ISO-2022-JP");
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvTo2022) {
    ASSERT_EQ(ConvTo2022.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }

  // Check that the string is properly converted.
  std::error_code EC = ConvTo2022->convert(Src, Dst, true);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(EarthISO2022, static_cast<std::string>(Dst).c_str());
}

TEST(CharSet, ShiftStateIBM939) {
  // Earth string.
  StringRef Src(EarthUTF);
  SmallString<64> Dst;

  ErrorOr<CharSetConverter> ConvToIBM939 =
      CharSetConverter::create("UTF-8", "IBM-939");
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvToIBM939) {
    ASSERT_EQ(ConvToIBM939.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }

  // Check that the string is properly converted.
  std::error_code EC = ConvToIBM939->convert(Src, Dst, true);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(EarthIBM939, static_cast<std::string>(Dst).c_str());
}

#if not defined(HAVE_ICU) && defined(HAVE_ICONV)

// Identical to EarthUTF, except the final character (球) has its last byte
// taken away from it.
static const char EarthUTFBroken[] = "\x45\x61\x72\x74\x68\xe5\x9c\xb0\xe7\x90";
static const char EarthISO2022ShiftBack[] =
    "\x45\x61\x72\x74\x68\x1B\x24\x42\x43\x4F\x35\x65";
static const char ShiftBackOnly[] = "\x1B\x28\x42";

// String "地球".
static const char EarthKanjiOnlyUTF[] = "\xe5\x9c\xb0\xe7\x90\x83";
static const char EarthKanjiOnlyISO2022[] =
    "\x1B\x24\x42\x43\x4F\x35\x65\x1b\x28\x42";
static const char EarthKanjiOnlyIBM939[] = "\x0e\x45\xc2\x48\xdb\x0f";

TEST(CharSet, ShiftState2022Flush) {
  StringRef Src0(EarthUTFBroken);
  StringRef Src1(EarthKanjiOnlyUTF);
  SmallString<64> Dst0;
  SmallString<64> Dst1;
  ErrorOr<CharSetConverter> ConvTo2022Flush =
      CharSetConverter::create("UTF-8", "ISO-2022-JP");
  if (!ConvTo2022Flush) {
    ASSERT_EQ(ConvTo2022Flush.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }

  // This should emit an error; there is a malformed multibyte character in the
  // input string.
  std::error_code EC0 = ConvTo2022Flush->convert(Src0, Dst0, true);
  EXPECT_TRUE(EC0);
  std::error_code EC1 = ConvTo2022Flush->flush();
  EXPECT_TRUE(!EC1);
  std::error_code EC2 = ConvTo2022Flush->convert(Src1, Dst1, true);
  EXPECT_TRUE(!EC2);
  EXPECT_STREQ(EarthKanjiOnlyISO2022, static_cast<std::string>(Dst1).c_str());
}

TEST(CharSet, ShiftStateIBM939Flush) {
  StringRef Src0(EarthUTFBroken);
  StringRef Src1(EarthKanjiOnlyUTF);
  SmallString<64> Dst0;
  SmallString<64> Dst1;
  ErrorOr<CharSetConverter> ConvTo939Flush =
      CharSetConverter::create("UTF-8", "IBM-939");
  if (!ConvTo939Flush) {
    ASSERT_EQ(ConvTo939Flush.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }

  // This should emit an error; there is a malformed multibyte character in the
  // input string.
  std::error_code EC0 = ConvTo939Flush->convert(Src0, Dst0, true);
  EXPECT_TRUE(EC0);
  std::error_code EC1 = ConvTo939Flush->flush();
  EXPECT_TRUE(!EC1);
  std::error_code EC2 = ConvTo939Flush->convert(Src1, Dst1, true);
  EXPECT_TRUE(!EC2);
  EXPECT_STREQ(EarthKanjiOnlyIBM939, static_cast<std::string>(Dst1).c_str());
}

TEST(CharSet, ShiftState2022Flush1) {
  StringRef Src0(EarthUTF);
  SmallString<64> Dst0;
  SmallString<64> Dst1;
  ErrorOr<CharSetConverter> ConvTo2022Flush =
      CharSetConverter::create("UTF-8", "ISO-2022-JP");
  if (!ConvTo2022Flush) {
    ASSERT_EQ(ConvTo2022Flush.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }

  std::error_code EC0 = ConvTo2022Flush->convert(Src0, Dst0, false);
  EXPECT_TRUE(!EC0);
  EXPECT_STREQ(EarthISO2022ShiftBack, static_cast<std::string>(Dst0).c_str());
  std::error_code EC1 = ConvTo2022Flush->flush(Dst1);
  EXPECT_TRUE(!EC1);
  EXPECT_STREQ(ShiftBackOnly, static_cast<std::string>(Dst1).c_str());
}

#endif

} // namespace
