//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/ConPTYUtils.h"
#include "gtest/gtest.h"
#include <cstring>
#include <string>

using namespace lldb_private;

namespace {

std::string Strip(const std::string &input, bool strip_init) {
  std::string buf(input);
  size_t len = buf.size();
  StripConPTYSequences(buf.data(), len, strip_init);
  return buf.substr(0, len);
}

} // namespace

TEST(StripConPTYSequencesTest, PassthroughPlainText) {
  EXPECT_EQ("Hello World!\r\n", Strip("Hello World!\r\n", true));
}

TEST(StripConPTYSequencesTest, EmptyInput) { EXPECT_EQ("", Strip("", true)); }

TEST(StripConPTYSequencesTest, StripCursorQuery) {
  EXPECT_EQ("", Strip("\x1b[6n", true));
  EXPECT_EQ("abc", Strip("abc\x1b[6n", true));
  EXPECT_EQ("abc", Strip("\x1b[6nabc", true));
}

TEST(StripConPTYSequencesTest, StripWin32InputMode) {
  EXPECT_EQ("", Strip("\x1b[?9001h", true));
  EXPECT_EQ("", Strip("\x1b[?9001l", true));
  EXPECT_EQ("abc", Strip("\x1b[?9001habc", true));
  EXPECT_EQ("abc", Strip("abc\x1b[?9001l", true));
}

TEST(StripConPTYSequencesTest, StripFocusEvents) {
  EXPECT_EQ("", Strip("\x1b[?1004h", true));
  EXPECT_EQ("", Strip("\x1b[?1004l", true));
  EXPECT_EQ("text", Strip("\x1b[?1004htext\x1b[?1004l", true));
}

TEST(StripConPTYSequencesTest, StripWindowTitle) {
  EXPECT_EQ("", Strip("\x1b]0;My Title\x07", true));
  EXPECT_EQ("abc", Strip("abc\x1b]0;window\x07", true));
  EXPECT_EQ("abc", Strip("\x1b]0;title\x07"
                         "abc",
                         true));
}

TEST(StripConPTYSequencesTest, WindowTitleWithoutBEL) {
  // If BEL is missing, strip to end of buffer.
  EXPECT_EQ("abc", Strip("abc\x1b]0;unterminated title", true));
}

TEST(StripConPTYSequencesTest, StripSGRResetOnlyWhenInit) {
  EXPECT_EQ("", Strip("\x1b[m", true));
  // With strip_init=false, \x1b[m is preserved.
  EXPECT_EQ("\x1b[m", Strip("\x1b[m", false));
}

TEST(StripConPTYSequencesTest, StripShowCursorOnlyWhenInit) {
  EXPECT_EQ("", Strip("\x1b[?25h", true));
  // With strip_init=false, \x1b[?25h is preserved.
  EXPECT_EQ("\x1b[?25h", Strip("\x1b[?25h", false));
}

TEST(StripConPTYSequencesTest, AlwaysStripNonInitSequences) {
  // Cursor query, Win32 Input Mode, focus events, and window title are
  // stripped regardless of strip_init.
  EXPECT_EQ("", Strip("\x1b[6n", false));
  EXPECT_EQ("", Strip("\x1b[?9001h", false));
  EXPECT_EQ("", Strip("\x1b[?1004l", false));
  EXPECT_EQ("", Strip("\x1b]0;title\x07", false));
}

TEST(StripConPTYSequencesTest, PreserveUnrecognizedEscape) {
  // An ESC sequence we don't recognize should be passed through.
  EXPECT_EQ("\x1b[31m", Strip("\x1b[31m", true));
  EXPECT_EQ("\x1b[H", Strip("\x1b[H", true));
}

TEST(StripConPTYSequencesTest, TypicalConPTYInitBurst) {
  // Simulates the first read from ConPTY with PSEUDOCONSOLE_INHERIT_CURSOR.
  std::string init = "\x1b[6n"            // cursor query
                     "\x1b[?9001h"        // Win32 Input Mode on
                     "\x1b[?1004h"        // focus events on
                     "\x1b[m"             // SGR reset
                     "\x1b]0;C:\\app\x07" // window title
                     "\x1b[?25h";         // show cursor
  EXPECT_EQ("", Strip(init, true));
}

TEST(StripConPTYSequencesTest, InitBurstWithInferiorOutput) {
  std::string input = "\x1b[6n\x1b[?9001h\x1b[?1004h\x1b[m"
                      "\x1b]0;app\x07\x1b[?25h"
                      "Hello World!\r\n";
  EXPECT_EQ("Hello World!\r\n", Strip(input, true));
}

TEST(StripConPTYSequencesTest, ShutdownSequences) {
  // ConPTY emits these when the process exits.
  std::string shutdown = "\x1b[?9001l\x1b[?1004l\r\n";
  EXPECT_EQ("\r\n", Strip(shutdown, false));
}

TEST(StripConPTYSequencesTest, MultipleSequencesInterleaved) {
  std::string input = "aaa\x1b[6nbbb\x1b[?9001hccc";
  EXPECT_EQ("aaabbbccc", Strip(input, true));
}

TEST(StripConPTYSequencesTest, PartialEscapeAtEnd) {
  // A lone ESC at the end of buffer - not a recognized sequence, pass through.
  std::string input = "text\x1b";
  EXPECT_EQ("text\x1b", Strip(input, true));
}

TEST(StripConPTYSequencesTest, LenUpdatedToZero) {
  std::string buf = "\x1b[6n";
  size_t len = buf.size();
  StripConPTYSequences(buf.data(), len, true);
  EXPECT_EQ(0u, len);
}
