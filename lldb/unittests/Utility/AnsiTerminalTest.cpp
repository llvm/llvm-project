//===-- AnsiTerminalTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/AnsiTerminal.h"

using namespace lldb_private;

TEST(AnsiTerminal, Empty) { EXPECT_EQ("", ansi::FormatAnsiTerminalCodes("")); }

TEST(AnsiTerminal, WhiteSpace) {
  EXPECT_EQ(" ", ansi::FormatAnsiTerminalCodes(" "));
  EXPECT_EQ(" ", ansi::StripAnsiTerminalCodes(" "));
}

TEST(AnsiTerminal, AtEnd) {
  EXPECT_EQ("abc\x1B[30m",
            ansi::FormatAnsiTerminalCodes("abc${ansi.fg.black}"));

  EXPECT_EQ("abc", ansi::StripAnsiTerminalCodes("abc\x1B[30m"));
}

TEST(AnsiTerminal, AtStart) {
  EXPECT_EQ("\x1B[30mabc",
            ansi::FormatAnsiTerminalCodes("${ansi.fg.black}abc"));

  EXPECT_EQ("abc", ansi::StripAnsiTerminalCodes("\x1B[30mabc"));
}

TEST(AnsiTerminal, KnownPrefix) {
  EXPECT_EQ("${ansi.fg.redish}abc",
            ansi::FormatAnsiTerminalCodes("${ansi.fg.redish}abc"));
}

TEST(AnsiTerminal, Unknown) {
  EXPECT_EQ("${ansi.fg.foo}abc",
            ansi::FormatAnsiTerminalCodes("${ansi.fg.foo}abc"));
}

TEST(AnsiTerminal, Incomplete) {
  EXPECT_EQ("abc${ansi.", ansi::FormatAnsiTerminalCodes("abc${ansi."));
}

TEST(AnsiTerminal, Twice) {
  EXPECT_EQ("\x1B[30m\x1B[31mabc",
            ansi::FormatAnsiTerminalCodes("${ansi.fg.black}${ansi.fg.red}abc"));

  EXPECT_EQ("abc", ansi::StripAnsiTerminalCodes("\x1B[30m\x1B[31mabc"));
}

TEST(AnsiTerminal, Basic) {
  EXPECT_EQ(
      "abc\x1B[31mabc\x1B[0mabc",
      ansi::FormatAnsiTerminalCodes("abc${ansi.fg.red}abc${ansi.normal}abc"));

  EXPECT_EQ("abcabcabc",
            ansi::StripAnsiTerminalCodes("abc\x1B[31mabc\x1B[0mabc"));
}

TEST(AnsiTerminal, InvalidEscapeCode) {
  EXPECT_EQ("abc\x1B[31kabcabc",
            ansi::StripAnsiTerminalCodes("abc\x1B[31kabc\x1B[0mabc"));
}
