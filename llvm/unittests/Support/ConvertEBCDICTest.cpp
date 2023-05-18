//===- unittests/Support/ConvertEBCDICTest.cpp - EBCDIC/UTF8 conversion tests
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------------===//

#include "llvm/Support/ConvertEBCDIC.h"
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

TEST(CharSet, FromUTF8) {
  // Hello string.
  StringRef Src(HelloA);
  SmallString<64> Dst;

  std::error_code EC = ConverterEBCDIC::convertToEBCDIC(Src, Dst);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(HelloE, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // ABC string.
  Src = ABCStrA;
  EC = ConverterEBCDIC::convertToEBCDIC(Src, Dst);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(ABCStrE, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // Accent string.
  Src = AccentUTF;
  EC = ConverterEBCDIC::convertToEBCDIC(Src, Dst);
  EXPECT_TRUE(!EC);
  EXPECT_STREQ(AccentE, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // Cyrillic string. Results in error because not representable in 1047.
  Src = CyrillicUTF;
  EC = ConverterEBCDIC::convertToEBCDIC(Src, Dst);
  EXPECT_EQ(EC, std::errc::illegal_byte_sequence);
  Dst.clear();
}

TEST(CharSet, ToUTF8) {
  // Hello string.
  StringRef Src(HelloE);
  SmallString<64> Dst;

  ConverterEBCDIC::convertToUTF8(Src, Dst);
  EXPECT_STREQ(HelloA, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // ABC string.
  Src = ABCStrE;
  ConverterEBCDIC::convertToUTF8(Src, Dst);
  EXPECT_STREQ(ABCStrA, static_cast<std::string>(Dst).c_str());
  Dst.clear();

  // Accent string.
  Src = AccentE;
  ConverterEBCDIC::convertToUTF8(Src, Dst);
  EXPECT_STREQ(AccentUTF, static_cast<std::string>(Dst).c_str());
  Dst.clear();
}

} // namespace
