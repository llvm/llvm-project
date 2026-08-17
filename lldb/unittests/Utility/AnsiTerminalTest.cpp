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

  // This string previously triggered a bug in handling incomplete Unicode
  // characters, when we had already accumulated some previous parts of the
  // string.
  const char *quick = "The \x1B[0mquick\x1B[0m 💨\x1B[0m";
  EXPECT_EQ(ansi::TrimAndPad(quick, 0), "");
  EXPECT_EQ(ansi::TrimAndPad(quick, 9), "The \x1B[0mquick\x1B[0m");
  EXPECT_EQ(ansi::TrimAndPad(quick, 10), "The \x1B[0mquick\x1B[0m ");
  // The emoji is 2 columns, so 11 is not quite enough.
  EXPECT_EQ(ansi::TrimAndPad(quick, 11, '_'), "The \x1B[0mquick\x1B[0m _");
  // 12 exactly enough to include the emoji and proceeding ANSI code.
  EXPECT_EQ(ansi::TrimAndPad(quick, 12), quick);
}

TEST(AnsiTerminal, TrimAtWordBoundary) {
  // Nothing in, nothing out.
  EXPECT_EQ(ansi::TrimAtWordBoundary("", 0), "");
  EXPECT_EQ(ansi::TrimAtWordBoundary("", 1), "");
  EXPECT_EQ(ansi::TrimAtWordBoundary("", 1), "");

  // All whitespace, return nothing.
  EXPECT_EQ(ansi::TrimAtWordBoundary("    ", 1), "");

  // Leading and trailing whitespace are removed.
  EXPECT_EQ(ansi::TrimAtWordBoundary("     ab     ", 0), "ab");
  EXPECT_EQ(ansi::TrimAtWordBoundary("     ab     ", 5), "ab");
  EXPECT_EQ(ansi::TrimAtWordBoundary("    🦊🦊     ", 0), "🦊🦊");
  EXPECT_EQ(ansi::TrimAtWordBoundary("    🦊🦊     ", 5), "🦊🦊");

  // When it is a single word, we ignore the max columns and return the word.
  EXPECT_EQ(ansi::TrimAtWordBoundary("abc", 0), "abc");
  EXPECT_EQ(ansi::TrimAtWordBoundary("abc", 1), "abc");
  EXPECT_EQ(ansi::TrimAtWordBoundary("abc", 2), "abc");
  EXPECT_EQ(ansi::TrimAtWordBoundary("abc", 3), "abc");
  EXPECT_EQ(ansi::TrimAtWordBoundary("abc", 4), "abc");
  EXPECT_EQ(ansi::TrimAtWordBoundary("abcdefghij", 2), "abcdefghij");
  EXPECT_EQ(ansi::TrimAtWordBoundary("🦊🦊", 0), "🦊🦊");
  EXPECT_EQ(ansi::TrimAtWordBoundary("🦊🦊", 4), "🦊🦊");

  // If it fits, return the entire word.
  EXPECT_EQ(ansi::TrimAtWordBoundary("abc", 5), "abc");
  EXPECT_EQ(ansi::TrimAtWordBoundary("🦊🦊", 5), "🦊🦊");

  // ANSI codes do not add to width.
  EXPECT_EQ(ansi::TrimAtWordBoundary("\x1B[0m", 0), "\x1B[0m");
  // Preceding ANSI codes are included.
  EXPECT_EQ(ansi::TrimAtWordBoundary("\x1B[0mab cd", 2), "\x1B[0mab");
  EXPECT_EQ(ansi::TrimAtWordBoundary("\x1B[0m🦊  🐱", 2), "\x1B[0m🦊");
  EXPECT_EQ(ansi::TrimAtWordBoundary("🦊\x1B[0m\x1B[0m🐱 🐈", 4),
            "🦊\x1B[0m\x1B[0m🐱");
  // If there's more than one, include all of them.
  EXPECT_EQ(ansi::TrimAtWordBoundary("\x1B[0m\x1B[0m\x1B[0mab cd", 2),
            "\x1B[0m\x1B[0m\x1B[0mab");
  // Proceeding ANSI codes are included.
  EXPECT_EQ(ansi::TrimAtWordBoundary("\x1B[0mab\x1B[0m cd", 2),
            "\x1B[0mab\x1B[0m");
  EXPECT_EQ(ansi::TrimAtWordBoundary("\x1B[0mab\x1B[0m", 4),
            "\x1B[0mab\x1B[0m");
  EXPECT_EQ(ansi::TrimAtWordBoundary("\x1B[0m🦊\x1B[0m 🐱", 2),
            "\x1B[0m🦊\x1B[0m");
  EXPECT_EQ(ansi::TrimAtWordBoundary("\x1B[0m🦊\x1B[0m", 4),
            "\x1B[0m🦊\x1B[0m");
  // Include all if more than one.
  EXPECT_EQ(ansi::TrimAtWordBoundary("\x1B[0mab\x1B[0m\x1B[0m\x1B[0m cd", 2),
            "\x1B[0mab\x1B[0m\x1B[0m\x1B[0m");
  // Mutliple pre and proceding ANSI codes.
  EXPECT_EQ(ansi::TrimAtWordBoundary("\x1B[0m\x1B[0mab\x1B[0m\x1B[0m cd", 2),
            "\x1B[0m\x1B[0mab\x1B[0m\x1B[0m");

  // When multiple words fit, include as many as we can while still ending on
  // a word boundary.
  const char *fox_ascii = "The quick brown fox jumped.";
  // Can't fit one word, just returns first word.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_ascii, 0), "The");
  // Exactly 3 is required for one word.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_ascii, 3), "The");
  // Exactly 9 is required to fit 2 words.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_ascii, 9), "The quick");
  // So anything less than 9 is just one word.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_ascii, 8), "The");
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_ascii, 4), "The");
  // 3 words is exactly 15.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_ascii, 15), "The quick brown");
  // Anything less is 2 words.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_ascii, 14), "The quick");
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_ascii, 10), "The quick");
  // The whole string.
  size_t fox_ascii_len = strlen(fox_ascii);
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_ascii, fox_ascii_len), fox_ascii);
  // Anything less and we remove the last word.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_ascii, fox_ascii_len - 1),
            "The quick brown fox");

  // Width calculation is Unicode aware and a run of Unicode is a word just
  // like a run of ASCII is.
  // Note that these emoji avoid any compound emoji where there are
  // non-printable modifiers. This is because llvm::sys::locale::columnWidth
  // returns -1 for these non-printable adjustment characters. At this time,
  // TrimAtWordBoundary simply cannot handle them well.
  const char *fox_unicode = "🦊 💨🟤 🔼";
  // Emoji have width 2, so this "word" would not fit so we just return it.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_unicode, 0), "🦊");
  // It does fit width 2.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_unicode, 2), "🦊");
  // Need 7 to fit 2 words.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_unicode, 7), "🦊 💨🟤");
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_unicode, 6), "🦊");
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_unicode, 4), "🦊");
  // The entire string.
  size_t fox_unicode_len = llvm::sys::locale::columnWidth(fox_unicode);
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_unicode, fox_unicode_len),
            "🦊 💨🟤 🔼");
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_unicode, fox_unicode_len - 1),
            "🦊 💨🟤");

  const char *fox_everything =
      "The \x1B[0mquick\x1B[0m 💨\x1B[0m brown \x1B[0m🟤 fox🦊 🔼jumped.";
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_everything, 0), "The");
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_everything, 3), "The");
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_everything, 6), "The");
  // Exactly 9 to fit two words.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_everything, 9),
            "The \x1B[0mquick\x1B[0m");
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_everything, 10),
            "The \x1B[0mquick\x1B[0m");
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_everything, 11),
            "The \x1B[0mquick\x1B[0m");
  // <space><2 wide emoji> adds 3 more to get to 12.
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_everything, 12),
            "The \x1B[0mquick\x1B[0m 💨\x1B[0m");
  // The entire string. We use the ansi:: width function here because it strips
  // ANSI codes that llvm::sys::locale's function cannot cope with.
  size_t fox_everything_len = ansi::ColumnWidth(fox_everything);
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_everything, fox_everything_len),
            fox_everything);
  EXPECT_EQ(ansi::TrimAtWordBoundary(fox_everything, fox_everything_len - 1),
            "The \x1B[0mquick\x1B[0m 💨\x1B[0m brown \x1B[0m🟤 fox🦊");
}

static void TestLines(const std::string &input, int indent,
                      uint32_t output_max_columns,
                      const llvm::StringRef &expected) {
  StreamString strm;
  strm.SetIndentLevel(indent);
  ansi::OutputWordWrappedLines(strm, input, output_max_columns,
                               /*use_color=*/false);
  EXPECT_EQ(expected, strm.GetString());
}

TEST(AnsiTerminal, OutputWordWrappedLines) {
  // Nothing in, nothing out. No newline, no indent.
  TestLines("", 0, 5, "");
  TestLines("", 5, 5, "");

  // A single line will have a newline on the end.
  TestLines("abc", 0, 1, "abc\n");
  TestLines("abc", 2, 5, "  abc\n");
  TestLines("🦊🦊", 0, 0, "🦊🦊\n");
  TestLines("🦊🦊", 0, 2, "🦊🦊\n");

  // If the indent uses up all the columns, print the word on the same line
  // anyway. This prevents us outputting indent only lines forever.
  TestLines("abcdefghij", 4, 2, "    abcdefghij\n");

  // Leading whitespace is ignored because we're going to indent using the
  // stream.
  TestLines("       ", 3, 10, "");
  TestLines("  abc", 0, 4, "abc\n");
  TestLines("        abc", 2, 6, "  abc\n");

  // Multiple lines. Each one ends with a newline.
  TestLines("abc def", 0, 4, "abc\ndef\n");
  TestLines("abc def", 0, 5, "abc\ndef\n");
  // Indent applied to each line.
  TestLines("abc def", 2, 4, "  abc\n  def\n");
  // First word is wider than a whole line, do not split that word.
  TestLines("aabbcc ddee", 0, 5, "aabbcc\nddee\n");

  const char *fox_str = "The quick brown fox.";
  TestLines(fox_str, 0, 30, "The quick brown fox.\n");
  TestLines(fox_str, 5, 30, "     The quick brown fox.\n");
  TestLines(fox_str, 2, 15, "  The quick\n  brown fox.\n");
  // Must remove the spaces from the end of the first line.
  TestLines("The quick       brown fox.", 0, 15, "The quick\nbrown fox.\n");

  // If ANSI formatting is applied to multiple words, that range of words may
  // be split over multiple lines.
  StreamString indented_strm;
  indented_strm.SetIndentLevel(2);
  ansi::OutputWordWrappedLines(indented_strm, "\x1B[4mabc def\x1B[0m ghi", 6,
                               /*use_color=*/false);
  // The two spaces before "def" would have the previous ANSI code applied to
  // them.
  EXPECT_EQ("  \x1B[4mabc\n  def\x1B[0m\n  ghi\n", indented_strm.GetString());

  // If we can emit ANSI, we can use cursor positions to skip forward,
  // which leaves the indent unformatted.
  // (in normal use the inputs are command descriptions, which already have
  // ANSI removed if the terminal does not support it)
  indented_strm.Clear();
  ansi::OutputWordWrappedLines(indented_strm, "\x1B[4mabc def\x1B[0m ghi", 6,
                               /*use_color=*/true);
  EXPECT_EQ("\x1B[2C\x1B[4mabc\n\x1B[2Cdef\x1B[0m\n\x1B[2Cghi\n",
            indented_strm.GetString());
}
