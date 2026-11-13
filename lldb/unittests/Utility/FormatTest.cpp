//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Format.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

TEST(FormatTest, NamesAndCharactersKeepTheirExistingMeaning) {
  Format format;
  ASSERT_TRUE(GetFormatFromCString("X", format));
  EXPECT_EQ(format, eFormatHexUppercase);
  ASSERT_TRUE(GetFormatFromCString("HeX", format));
  EXPECT_EQ(format, eFormatHex);
  ASSERT_TRUE(GetFormatFromCString("unsigned dec", format));
  EXPECT_EQ(format, eFormatUnsigned);
  ASSERT_TRUE(GetFormatFromCString("u", format));
  EXPECT_EQ(format, eFormatUnsigned);
  ASSERT_TRUE(GetFormatFromCString("unicode8", format));
  EXPECT_EQ(format, eFormatUnicode8);
  EXPECT_FALSE(GetFormatFromCString("invalid format", format));
  EXPECT_EQ(format, eFormatInvalid);
  EXPECT_FALSE(GetFormatFromCString(nullptr, format));
  EXPECT_EQ(format, eFormatInvalid);
  EXPECT_FALSE(GetFormatFromCString("", format));
  EXPECT_EQ(format, eFormatInvalid);
}

TEST(FormatTest, EveryFormatHasItsEstablishedName) {
  for (unsigned value = eFormatDefault; value < kNumFormats; ++value) {
    Format format = static_cast<Format>(value);
    const char *name = GetFormatAsCString(format);
    ASSERT_NE(name, nullptr);
    Format parsed;
    ASSERT_TRUE(GetFormatFromCString(name, parsed));
    EXPECT_EQ(parsed, format);
  }
  EXPECT_EQ(GetFormatAsFormatChar(eFormatHex), 'x');
  // eFormatInvalid is an alias of eFormatDefault.
  EXPECT_STREQ(GetFormatAsCString(eFormatInvalid), "default");
}
