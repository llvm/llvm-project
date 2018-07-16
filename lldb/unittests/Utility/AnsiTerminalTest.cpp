//===-- AnsiTerminalTest.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/AnsiTerminal.h"

using namespace lldb_utility;

TEST(AnsiTerminal, Empty) { EXPECT_EQ("", ansi::FormatAnsiTerminalCodes("")); }

TEST(AnsiTerminal, WhiteSpace) {
  EXPECT_EQ(" ", ansi::FormatAnsiTerminalCodes(" "));
}

TEST(AnsiTerminal, AtEnd) {
  EXPECT_EQ("abc\x1B[30m",
            ansi::FormatAnsiTerminalCodes("abc${ansi.fg.black}"));
}

TEST(AnsiTerminal, AtStart) {
  EXPECT_EQ("\x1B[30mabc",
            ansi::FormatAnsiTerminalCodes("${ansi.fg.black}abc"));
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
}

TEST(AnsiTerminal, Basic) {
  EXPECT_EQ(
      "abc\x1B[31mabc\x1B[0mabc",
      ansi::FormatAnsiTerminalCodes("abc${ansi.fg.red}abc${ansi.normal}abc"));
}
