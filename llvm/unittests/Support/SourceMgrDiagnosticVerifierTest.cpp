//===- unittests/Support/SourceMgrDiagnosticVerifierTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SourceMgrDiagnosticVerifier.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using MatchResult = SourceMgrDiagnosticVerifier::MatchResult;
using testing::HasSubstr;

namespace {

class SourceMgrDiagnosticVerifierTest : public testing::Test {
protected:
  SourceMgr SM;
  std::string Output;
  raw_string_ostream OS{Output};
  SourceMgrDiagnosticVerifier Verifier;

  const MemoryBuffer *addBuffer(StringRef Text, StringRef BufferName) {
    unsigned ID = SM.AddNewSourceBuffer(
        MemoryBuffer::getMemBuffer(Text, BufferName), SMLoc());
    return SM.getMemoryBuffer(ID);
  }

  MatchResult processError(const MemoryBuffer *Buf, unsigned LineNo,
                           StringRef Message) {
    return Verifier.process(OS, SM, SourceMgr::DK_Error, /*hasLoc=*/true, Buf,
                            LineNo, Message);
  }
};

} // namespace

TEST_F(SourceMgrDiagnosticVerifierTest, BuffersWithSameName) {
  const MemoryBuffer *Buf1 =
      addBuffer("op // expected-error {{first}}\n", "same.s");
  const MemoryBuffer *Buf2 =
      addBuffer("op // expected-error {{second}}\n", "same.s");

  // Directives only match diagnostics in their own buffer.
  EXPECT_EQ(processError(Buf2, 1, "first"), MatchResult::Unexpected);
  // FIXME: Buf1's directives are never scanned since Buf2 has the same name.
  EXPECT_EQ(processError(Buf1, 1, "first"), MatchResult::Unexpected);
  EXPECT_EQ(processError(Buf2, 1, "second"), MatchResult::Matched);
}

TEST_F(SourceMgrDiagnosticVerifierTest, BufferAddedLater) {
  // E.g. an included file without any diagnostics.
  addBuffer("op // expected-error {{missing}}\n", "inc.s");

  // FIXME: Buffers without diagnostics are never scanned, so their directives
  // are not checked.
  EXPECT_TRUE(Verifier.verify(OS, SM));
  EXPECT_EQ(Output, "");
}

TEST_F(SourceMgrDiagnosticVerifierTest, OctalLineOffset) {
  const MemoryBuffer *Buf = addBuffer(
      "op\n\n\n\n\n\n\n\n\n\n// expected-error@-010 {{decimal}}\n", "test.s");
  // FIXME: '010' is parsed as octal.
  EXPECT_EQ(processError(Buf, 3, "decimal"), MatchResult::Matched);
}

// FIXME: '08' is not valid octal, so the offset is read uninitialized.
TEST_F(SourceMgrDiagnosticVerifierTest, DISABLED_LeadingZeroLineOffset) {
  const MemoryBuffer *Buf = addBuffer(
      "op\n\n\n\n\n\n\n\n// expected-error@-08 {{decimal}}\n", "test.s");
  EXPECT_EQ(processError(Buf, 1, "decimal"), MatchResult::Matched);
}

// FIXME: The offset does not fit in an int, so it is read uninitialized.
TEST_F(SourceMgrDiagnosticVerifierTest, DISABLED_OverflowingLineOffset) {
  const MemoryBuffer *Buf =
      addBuffer("// expected-error@+99999999999 {{overflow}}\n", "test.s");
  (void)Verifier.computeExpectedDiags(OS, SM, Buf);
  EXPECT_THAT(
      Output,
      HasSubstr("test.s:1:4: error: invalid line offset '+99999999999'"));
}

TEST_F(SourceMgrDiagnosticVerifierTest, OutOfRangeLineOffsets) {
  const MemoryBuffer *Buf = addBuffer("// expected-error@-1 {{line zero}}\n"
                                      "// expected-error@-3 {{before start}}\n"
                                      "// expected-error@+99 {{past end}}\n",
                                      "test.s");
  (void)Verifier.computeExpectedDiags(OS, SM, Buf);
  EXPECT_FALSE(Verifier.verify(OS, SM));
  // Line 0 is allowed since MLIR uses it for the implicit top-level module.
  EXPECT_THAT(Output,
              HasSubstr("test.s:1:4: error: expected error \"line zero\" "
                        "was not produced"));
  // FIXME: Offsets that point outside the file are not rejected.
  EXPECT_THAT(Output, HasSubstr("test.s:2:4: error: expected error \"before "
                                "start\" was not produced"));
  EXPECT_THAT(Output,
              HasSubstr("test.s:3:4: error: expected error \"past end\" "
                        "was not produced"));
}
