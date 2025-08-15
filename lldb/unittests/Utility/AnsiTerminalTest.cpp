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

TEST(AnsiTerminal, FindNextAnsiSequenceBasic) {
  auto [left, escape, right] = ansi::FindNextAnsiSequence("foo\x1B[31mbar");
  EXPECT_EQ("foo", left);
  EXPECT_EQ("\x1B[31m", escape);
  EXPECT_EQ("bar", right);
}

TEST(AnsiTerminal, FindNextAnsiSequenceIncompleteStart) {
  auto [left, escape, right] =
      ansi::FindNextAnsiSequence("foo\x1B[bar\x1B[31mbaz");
  EXPECT_EQ("foo\x1B[bar", left);
  EXPECT_EQ("\x1B[31m", escape);
  EXPECT_EQ("baz", right);
}

TEST(AnsiTerminal, FindNextAnsiSequenceEscapeStart) {
  auto [left, escape, right] = ansi::FindNextAnsiSequence("\x1B[31mfoo");
  EXPECT_EQ("", left);
  EXPECT_EQ("\x1B[31m", escape);
  EXPECT_EQ("foo", right);
}

TEST(AnsiTerminal, TrimAndPad) {
  // Test basic ASCII.
  EXPECT_EQ("     ", ansi::TrimAndPad("", 5));
  EXPECT_EQ("foo  ", ansi::TrimAndPad("foo", 5));
  EXPECT_EQ("fooba", ansi::TrimAndPad("fooba", 5));
  EXPECT_EQ("fooba", ansi::TrimAndPad("foobar", 5));

  // Simple test that ANSI escape codes don't contribute to the visible width.
  EXPECT_EQ("\x1B[30m     ", ansi::TrimAndPad("\x1B[30m", 5));
  EXPECT_EQ("\x1B[30mfoo  ", ansi::TrimAndPad("\x1B[30mfoo", 5));
  EXPECT_EQ("\x1B[30mfooba", ansi::TrimAndPad("\x1B[30mfooba", 5));
  EXPECT_EQ("\x1B[30mfooba", ansi::TrimAndPad("\x1B[30mfoobar", 5));

  // Test that we include as many escape codes as we can.
  EXPECT_EQ("fooba\x1B[30m", ansi::TrimAndPad("fooba\x1B[30m", 5));
  EXPECT_EQ("fooba\x1B[30m\x1B[34m",
            ansi::TrimAndPad("fooba\x1B[30m\x1B[34m", 5));
  EXPECT_EQ("fooba\x1B[30m\x1B[34m",
            ansi::TrimAndPad("fooba\x1B[30m\x1B[34mr", 5));

  // Test Unicode.
  EXPECT_EQ("❤️    ", ansi::TrimAndPad("❤️", 5));
  EXPECT_EQ("    ❤️", ansi::TrimAndPad("    ❤️", 5));
  EXPECT_EQ("12❤️4❤️", ansi::TrimAndPad("12❤️4❤️", 5));
  EXPECT_EQ("12❤️45", ansi::TrimAndPad("12❤️45❤️", 5));
}
