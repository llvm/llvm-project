//===- unittest/Support/RemarksStrTabParsingTest.cpp - StrTab tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkParser.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(RemarksStrTab, ParsingEmpty) {
  StringRef Empty("", 0);
  remarks::ParsedStringTable StrTab(Empty);
  Expected<StringRef> Nothing = StrTab[0];
  EXPECT_FALSE(static_cast<bool>(Nothing));
  EXPECT_EQ(toString(Nothing.takeError()),
            "String with index 0 is out of bounds (size = 0).");
}

TEST(RemarksStrTab, ParsingGood) {
  StringRef Strings("str1\0str2\0str3\0str4", 20);
  remarks::ParsedStringTable StrTab(Strings);
  Expected<StringRef> Result = StrTab[0];
  EXPECT_TRUE(static_cast<bool>(Result));
  EXPECT_EQ(*Result, "str1");
  Result = StrTab[1];
  EXPECT_TRUE(static_cast<bool>(Result));
  EXPECT_EQ(*Result, "str2");
  Result = StrTab[2];
  EXPECT_TRUE(static_cast<bool>(Result));
  EXPECT_EQ(*Result, "str3");
  Result = StrTab[3];
  EXPECT_TRUE(static_cast<bool>(Result));
  EXPECT_EQ(*Result, "str4");
}
