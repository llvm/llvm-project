//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TextEncoding.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/config.h"
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
static const char EarthISO2022[] =
    "\x45\x61\x72\x74\x68\x1B\x24\x42\x43\x4F\x35\x65\x1B\x28\x42";
static const char EarthIBM939[] =
    "\xc5\x81\x99\xa3\x88\x0e\x45\xc2\x48\xdb\x0f";
static const char EarthUTFExtraPartial[] =
    "\x45\x61\x72\x74\x68\xe5\x9c\xb0\xe7\x90\x83\xe5";

TEST(Encoding, FromUTF8) {
  // Hello string.
  StringRef Src(HelloA);
  SmallString<64> Dst;

  ErrorOr<TextEncodingConverter> Conv =
      TextEncodingConverter::create(TextEncoding::UTF8, TextEncoding::IBM1047);

  // Converter should always exist between UTF-8 and IBM-1047
  EXPECT_TRUE(Conv);

  std::error_code EC = Conv->convert(Src, Dst);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(HelloE, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // ABC string.
  Src = ABCStrA;
  EC = Conv->convert(Src, Dst);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(ABCStrE, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // Accent string.
  Src = AccentUTF;
  EC = Conv->convert(Src, Dst);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(AccentE, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // Cyrillic string. Results in error because not representable in 1047.
  Src = CyrillicUTF;
  EC = Conv->convert(Src, Dst);
  EXPECT_EQ(EC, std::errc::illegal_byte_sequence);
}

TEST(Encoding, ToUTF8) {
  // Hello string.
  StringRef Src(HelloE);
  SmallString<64> Dst;

  ErrorOr<TextEncodingConverter> Conv =
      TextEncodingConverter::create(TextEncoding::IBM1047, TextEncoding::UTF8);

  // Converter should always exist between UTF-8 and IBM-1047
  EXPECT_TRUE(Conv);

  std::error_code EC = Conv->convert(Src, Dst);

  EXPECT_TRUE(!EC);
  EXPECT_STREQ(HelloA, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // ABC string.
  Src = ABCStrE;
  EC = Conv->convert(Src, Dst);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(ABCStrA, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // Accent string.
  Src = AccentE;
  EC = Conv->convert(Src, Dst);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(AccentUTF, static_cast<std::string>(Dst).c_str());
}

TEST(Encoding, RoundTrip) {
  ErrorOr<TextEncodingConverter> ConvToUTF16 =
      TextEncodingConverter::create("IBM-1047", "UTF-16");

#if HAVE_ICU
  EXPECT_TRUE(ConvToUTF16);
#else
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvToUTF16) {
    ASSERT_EQ(ConvToUTF16.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }
#endif

  ErrorOr<TextEncodingConverter> ConvToUTF32 =
      TextEncodingConverter::create("UTF-16", "UTF-32");

#if HAVE_ICU
  EXPECT_TRUE(ConvToUTF32);
#else
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvToUTF32) {
    ASSERT_EQ(ConvToUTF32.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }
#endif

  ErrorOr<TextEncodingConverter> ConvToEBCDIC =
      TextEncodingConverter::create("UTF-32", "IBM-1047");

#if HAVE_ICU
  EXPECT_TRUE(ConvToEBCDIC);
#else
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvToEBCDIC) {
    ASSERT_EQ(ConvToEBCDIC.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }
#endif

  // Setup source string.
  char SrcStr[256];
  for (size_t I = 0; I < 256; ++I)
    SrcStr[I] = (I + 1) % 256;

  SmallString<99> Dst1Str, Dst2Str, Dst3Str;

  std::error_code EC = ConvToUTF16->convert(StringRef(SrcStr), Dst1Str);
  EXPECT_TRUE(!EC);
  EC = ConvToUTF32->convert(Dst1Str, Dst2Str);
  EXPECT_TRUE(!EC);
  EC = ConvToEBCDIC->convert(Dst2Str, Dst3Str);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(SrcStr, static_cast<std::string>(Dst3Str).c_str());
}

TEST(Encoding, ShiftState2022) {
  // Earth string.
  StringRef Src(EarthUTF);
  SmallString<8> Dst;

  ErrorOr<TextEncodingConverter> ConvTo2022 =
      TextEncodingConverter::create("UTF-8", "ISO-2022-JP");

#if HAVE_ICU
  EXPECT_TRUE(ConvTo2022);
#else
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvTo2022) {
    ASSERT_EQ(ConvTo2022.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }
#endif

  // Check that the string is properly converted.
  std::error_code EC = ConvTo2022->convert(Src, Dst);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(EarthISO2022, static_cast<std::string>(Dst).c_str());
}

TEST(Encoding, InvalidInput) {
  // Earth string.
  StringRef Src(EarthUTFExtraPartial);
  SmallString<8> Dst;

  ErrorOr<TextEncodingConverter> ConvTo2022 =
      TextEncodingConverter::create("UTF-8", "ISO-2022-JP");

#if HAVE_ICU
  EXPECT_TRUE(ConvTo2022);
#else
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvTo2022) {
    ASSERT_EQ(ConvTo2022.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }
#endif

  // Check that the string failed to convert.
  std::error_code EC = ConvTo2022->convert(Src, Dst);
  EXPECT_TRUE(EC);
}

TEST(Encoding, InvalidOutput) {
  // Cyrillic in UTF-16
  ErrorOr<TextEncodingConverter> ConvToUTF16 =
      TextEncodingConverter::create("UTF-8", "UTF-16");

#if HAVE_ICU
  EXPECT_TRUE(ConvToUTF16);
#else
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvToUTF16) {
    ASSERT_EQ(ConvToUTF16.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }
#endif

  ErrorOr<TextEncodingConverter> ConvToEBCDIC =
      TextEncodingConverter::create("UTF-16", "IBM-1047");

#if HAVE_ICU
  EXPECT_TRUE(ConvToEBCDIC);
#else
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvToEBCDIC) {
    ASSERT_EQ(ConvToEBCDIC.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }
#endif

  // Cyrillic string. Convert to UTF-16 and check if properly converted
  StringRef Src(CyrillicUTF);
  SmallString<8> Dst, Dst1;
  std::error_code EC = ConvToUTF16->convert(Src, Dst);
  EXPECT_TRUE(!EC);

  // Cyrillic string. Results in error because not representable in 1047.
  EC = ConvToEBCDIC->convert(Dst, Dst1);
  EXPECT_TRUE(EC);
}

TEST(Encoding, ShiftStateIBM939) {
  // Earth string.
  StringRef Src(EarthUTF);
  SmallString<64> Dst;

  ErrorOr<TextEncodingConverter> ConvToIBM939 =
      TextEncodingConverter::create("UTF-8", "IBM-939");

#if HAVE_ICU
  EXPECT_TRUE(ConvToIBM939);
#else
  // Stop test if conversion is not supported (no underlying iconv support).
  if (!ConvToIBM939) {
    ASSERT_EQ(ConvToIBM939.getError(),
              std::make_error_code(std::errc::invalid_argument));
    return;
  }
#endif

  // Check that the string is properly converted.
  std::error_code EC = ConvToIBM939->convert(Src, Dst);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(EarthIBM939, static_cast<std::string>(Dst).c_str());
}

} // namespace
