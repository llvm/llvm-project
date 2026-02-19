//===-- AnsiTerminalTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/StreamString.h"

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

static void TestLines(const std::string &input, int indent,
                      uint32_t output_max_columns,
                      const llvm::StringRef &expected) {
  StreamString strm;
  strm.SetIndentLevel(indent);
  ansi::OutputWordWrappedLines(strm, input, output_max_columns);
  EXPECT_EQ(expected, strm.GetString());
}

TEST(AnsiTerminal, OutputWordWrappedLines) {
  TestLines("", 0, 0, "");
  TestLines("", 0, 1, "");
  TestLines("", 2, 1, "");

  // When it is a single word, we ignore the max columns and do not split it.
  TestLines("abc", 0, 1, "abc\n");
  TestLines("abc", 0, 2, "abc\n");
  TestLines("abc", 0, 3, "abc\n");
  TestLines("abc", 0, 4, "abc\n");
  TestLines("abc", 1, 5, " abc\n");
  TestLines("abc", 2, 5, "  abc\n");
  // If the indent uses up all the columns, print the word on the same line
  // anyway. This prevents us outputting indent only lines forever.
  TestLines("abcdefghij", 4, 2, "    abcdefghij\n");

  // Leading whitespace is ignored because we're going to indent using the
  // stream.
  TestLines("       ", 3, 10, "");
  TestLines("  abc", 0, 4, "abc\n");
  TestLines("        abc", 2, 6, "  abc\n");

  TestLines("abc def", 0, 4, "abc\ndef\n");
  TestLines("abc def", 0, 5, "abc\ndef\n");
  // Length is 6, 7 required. Has to split at whitespace.
  TestLines("abc def", 0, 6, "abc\ndef\n");
  // FIXME: This should split after abc, and not print
  // more whitespace on the end of the line or the start
  // of the new one. Resulting in "abc\ndef\n".
  TestLines("abc           def", 0, 6, "abc  \ndef\n");

  const char *fox_str = "The quick brown fox.";
  TestLines(fox_str, 0, 30, "The quick brown fox.\n");
  TestLines(fox_str, 5, 30, "     The quick brown fox.\n");
  TestLines(fox_str, 0, 15, "The quick\nbrown fox.\n");
  // FIXME: Trim the spaces off of the end of the first line.
  TestLines("The quick       brown fox.", 0, 15,
            "The quick     \nbrown fox.\n");

  // As ANSI codes do not add to visible length, the results
  // should be the same as the plain text version.
  const char *fox_str_ansi = "\x1B[4mT\x1B[0mhe quick brown fox.";
  TestLines(fox_str_ansi, 0, 30, "\x1B[4mT\x1B[0mhe quick brown fox.\n");
  TestLines(fox_str_ansi, 5, 30, "     \x1B[4mT\x1B[0mhe quick brown fox.\n");
  TestLines(fox_str_ansi, 0, 15, "\x1B[4mT\x1B[0mhe quick\nbrown fox.\n");

  // This should be the same when it applies to >1 character.
  const char *fox_str_ansi_2 = "\x1B[4mTh\x1B[0me quick brown fox.";
  TestLines(fox_str_ansi_2, 0, 30, "\x1B[4mTh\x1B[0me quick brown fox.\n");
  TestLines(fox_str_ansi_2, 5, 30, "     \x1B[4mTh\x1B[0me quick brown fox.\n");
  TestLines(fox_str_ansi_2, 0, 15, "\x1B[4mTh\x1B[0me quick\nbrown fox.\n");

  // Or when the ANSI code is at the end of the string.
  const char *fox_str_ansi_3 = "The quick brown fox\x1B[4m.\x1B[0m";
  TestLines(fox_str_ansi_3, 0, 30, "The quick brown fox\x1B[4m.\x1B[0m\n");
  TestLines(fox_str_ansi_3, 5, 30, "     The quick brown fox\x1B[4m.\x1B[0m\n");
  TestLines(fox_str_ansi_3, 0, 15, "The quick\nbrown fox\x1B[4m.\x1B[0m\n");

  // FIXME: ANSI codes applied to > 1 word end up applying to all those words
  // and the indent if those words are split up. We should use cursor
  // positioning to do the indentation instead.
  TestLines("\x1B[4mabc def\x1B[0m ghi", 2, 6,
            "  \x1B[4mabc\n  def\x1B[0m\n  ghi\n");

  const std::string fox_str_emoji = "🦊 The quick brown fox. 🦊";
  TestLines(fox_str_emoji, 0, 30, "🦊 The quick brown fox. 🦊\n");
  // FIXME: This crashes when max columns is exactly 31.
  // TestLines(fox_str_emoji, 5, 31, "     🦊 The quick brown fox. 🦊\n");
  TestLines(fox_str_emoji, 5, 32, "     🦊 The quick brown fox. 🦊\n");
  // FIXME: Final fox is missing.
  TestLines(fox_str_emoji, 0, 15, "🦊 The quick\nbrown fox. \n");
  // FIXME: should not split the middle of an emoji.
  TestLines("🦊🦊🦊 🦊🦊", 0, 5, "\n\n\n\n\n\n\n\x8A\xF0\x9F\xA6\n");
}

TEST(AnsiTerminal, VisiblePositionToActualPosition) {
  using Hint = ansi::VisibleActualPositionPair;
  EXPECT_EQ((Hint{0, 0}),
            ansi::VisiblePositionToActualPosition("", 0, std::nullopt));
  EXPECT_EQ((Hint{0, 0}),
            ansi::VisiblePositionToActualPosition("abc", 0, std::nullopt));
  EXPECT_EQ((Hint{1, 1}),
            ansi::VisiblePositionToActualPosition("abc", 1, std::nullopt));
  EXPECT_EQ((Hint{2, 2}),
            ansi::VisiblePositionToActualPosition("abc", 2, std::nullopt));
  // We expect callers to limit the visible index to its valid range.

  // When a visible character is preceeded by an ANSI code, we would need to
  // print that code when printing the character. Therefore, the actual index
  // points to the preceeding ANSI code, not the visible character itself.
  EXPECT_EQ((Hint{0, 0}), ansi::VisiblePositionToActualPosition("\x1B[4mabc", 0,
                                                                std::nullopt));
  EXPECT_EQ((Hint{1, 1}), ansi::VisiblePositionToActualPosition("a\x1B[4mbc", 1,
                                                                std::nullopt));
  EXPECT_EQ((Hint{2, 2}), ansi::VisiblePositionToActualPosition("ab\x1B[4mc", 2,
                                                                std::nullopt));

  // If the visible character is proceeded by an ANSI code, we don't need to
  // adjust anything. The actual index is the index of the visible character
  // itself.
  EXPECT_EQ((Hint{0, 0}), ansi::VisiblePositionToActualPosition("a\x1B[4mbc", 0,
                                                                std::nullopt));
  EXPECT_EQ((Hint{1, 1}), ansi::VisiblePositionToActualPosition("ab\x1B[4mc", 1,
                                                                std::nullopt));
  EXPECT_EQ((Hint{2, 2}), ansi::VisiblePositionToActualPosition("abc\x1B[4m", 2,
                                                                std::nullopt));

  // If we want a visible character that is after ANSI codes for other
  // characters, the actual index must know to skip those previous codes.
  EXPECT_EQ((Hint{1, 5}), ansi::VisiblePositionToActualPosition("\x1B[4mabc", 1,
                                                                std::nullopt));
  EXPECT_EQ((Hint{2, 6}), ansi::VisiblePositionToActualPosition("a\x1B[4mbc", 2,
                                                                std::nullopt));

  // We can give it a previous result to skip forward. To prove it does not look
  // at the early parts of the string, give it hints that actually produce
  // incorrect results.

  const char *actual_text = "abcdefghijk";
  // This does nothing because the hint is the answer.
  EXPECT_EQ((Hint{1, 5}),
            ansi::VisiblePositionToActualPosition(actual_text, 1, Hint{1, 5}));
  // The hint can be completely bogus, but we trust it. So if it refers to the
  // visible index being asked for, we just return the hint.
  EXPECT_EQ((Hint{99, 127}), ansi::VisiblePositionToActualPosition(
                                 actual_text, 99, Hint{99, 127}));
  // This should return {2, 1}, but the actual is offset by 5 due to the hint.
  EXPECT_EQ((Hint{2, 6}),
            ansi::VisiblePositionToActualPosition(actual_text, 2, Hint{1, 5}));
  // If the hint is for a visible index > the wanted visible index, we cannot do
  // anything with it. The function can only look forward.
  EXPECT_EQ((Hint{2, 2}),
            ansi::VisiblePositionToActualPosition(actual_text, 2, Hint{3, 6}));
}
