//===- unittests/Rewrite/RewriteBufferTest.cpp - RewriteBuffer tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/RewriteBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

#define EXPECT_OUTPUT(Buf, Output) EXPECT_EQ(Output, writeOutput(Buf))

static std::string writeOutput(const RewriteBuffer &Buf) {
  std::string Result;
  raw_string_ostream OS(Result);
  Buf.write(OS);
  OS.flush();
  return Result;
}

static void tagRange(unsigned Offset, unsigned Len, StringRef tagName,
                     RewriteBuffer &Buf) {
  std::string BeginTag;
  raw_string_ostream(BeginTag) << '<' << tagName << '>';
  std::string EndTag;
  raw_string_ostream(EndTag) << "</" << tagName << '>';

  Buf.InsertTextAfter(Offset, BeginTag);
  Buf.InsertTextBefore(Offset + Len, EndTag);
}

TEST(RewriteBuffer, TagRanges) {
  StringRef Input = "hello world";
  const char *Output = "<outer><inner>hello</inner></outer> ";

  RewriteBuffer Buf;
  Buf.Initialize(Input);
  StringRef RemoveStr = "world";
  size_t Pos = Input.find(RemoveStr);
  Buf.RemoveText(Pos, RemoveStr.size());

  StringRef TagStr = "hello";
  Pos = Input.find(TagStr);
  tagRange(Pos, TagStr.size(), "outer", Buf);
  tagRange(Pos, TagStr.size(), "inner", Buf);

  EXPECT_OUTPUT(Buf, Output);
}

TEST(RewriteBuffer, DISABLED_RemoveLineIfEmpty_XFAIL) {
  StringRef Input = "def\n"
                    "ghi\n"
                    "jkl\n";
  RewriteBuffer Buf;
  Buf.Initialize(Input);

  // Insert "abc\n" at the start.
  Buf.InsertText(0, "abc\n");
  EXPECT_OUTPUT(Buf, "abc\n"
                     "def\n"
                     "ghi\n"
                     "jkl\n");

  // Remove "def\n".
  //
  // After the removal of "def", we have:
  //
  //   "abc\n"
  //   "\n"
  //   "ghi\n"
  //   "jkl\n"
  //
  // Because removeLineIfEmpty=true, RemoveText has to remove the "\n" left on
  // the line.  This happens correctly for the rewrite buffer itself, so the
  // next check below passes.
  //
  // However, RemoveText's implementation incorrectly records the delta for
  // removing the "\n" using the rewrite buffer offset, 4, where it was
  // supposed to use the original input offset, 3.  Interpreted as an original
  // input offset, 4 points to "g" not to "\n".  Thus, any future modifications
  // at the original input's "g" will incorrectly see "g" as having become an
  // empty string and so will map to the next character, "h", in the rewrite
  // buffer.
  StringRef RemoveStr0 = "def";
  Buf.RemoveText(Input.find(RemoveStr0), RemoveStr0.size(),
                 /*removeLineIfEmpty*/ true);
  EXPECT_OUTPUT(Buf, "abc\n"
                     "ghi\n"
                     "jkl\n");

  // Try to remove "ghi\n".
  //
  // As discussed above, the original input offset for "ghi\n" incorrectly
  // maps to the rewrite buffer offset for "hi\nj", so we end up with:
  //
  //   "abc\n"
  //   "gkl\n"
  //
  // To show that removeLineIfEmpty=true is the culprit, change true to false
  // and append a newline to RemoveStr0 above.  The test then passes.
  StringRef RemoveStr1 = "ghi\n";
  Buf.RemoveText(Input.find(RemoveStr1), RemoveStr1.size());
  EXPECT_OUTPUT(Buf, "abc\n"
                     "jkl\n");
}

} // anonymous namespace
