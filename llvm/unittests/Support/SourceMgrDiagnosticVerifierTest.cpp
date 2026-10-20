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
#include "gtest/gtest.h"

using namespace llvm;
using MatchResult = SourceMgrDiagnosticVerifier::MatchResult;

namespace {

class SourceMgrDiagnosticVerifierTest : public testing::Test {
protected:
  SourceMgr SM;
  std::string Output;
  raw_string_ostream OS{Output};

  const MemoryBuffer *addBuffer(StringRef Text,
                                StringRef BufferName = "test.s") {
    unsigned ID = SM.AddNewSourceBuffer(
        MemoryBuffer::getMemBuffer(Text, BufferName), SMLoc());
    return SM.getMemoryBuffer(ID);
  }

  SMDiagnostic getMessage(const MemoryBuffer *Buf, SourceMgr::DiagKind Kind,
                          StringRef Msg) {
    return SM.GetMessage(SMLoc::getFromPointer(Buf->getBufferStart()), Kind,
                         Msg);
  }
};

} // namespace

TEST_F(SourceMgrDiagnosticVerifierTest, Match) {
  const MemoryBuffer *Buf =
      addBuffer("bad_op\n# expected-error@-1 {{invalid operand}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  EXPECT_EQ(
      Verifier.process(getMessage(Buf, SourceMgr::DK_Error, "invalid operand")),
      MatchResult::Matched);
  EXPECT_TRUE(Verifier.verify());
  EXPECT_EQ(Output, "");
}

TEST_F(SourceMgrDiagnosticVerifierTest, UnknownLocation) {
  addBuffer("# expected-warning@unknown {{no location warning}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  SMDiagnostic Diag =
      SM.GetMessage(SMLoc(), SourceMgr::DK_Warning, "no location warning");
  EXPECT_EQ(Verifier.process(Diag), MatchResult::Matched);
  EXPECT_TRUE(Verifier.verify());
  EXPECT_EQ(Output, "");
}

TEST_F(SourceMgrDiagnosticVerifierTest, UnmatchedUnknownLocation) {
  addBuffer("# expected-error@unknown {{missing unknown error}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  EXPECT_FALSE(Verifier.verify());
  EXPECT_NE(
      StringRef(Output).find(
          "error: expected error \"missing unknown error\" was not produced"),
      StringRef::npos);
}

TEST_F(SourceMgrDiagnosticVerifierTest, ReportUnexpectedFalse) {
  const MemoryBuffer *Buf =
      addBuffer("op\n# expected-error@-1 {{expected msg}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  // An unexpected diagnostic with ReportUnexpected = false is ignored and does
  // not fail verification.
  EXPECT_EQ(
      Verifier.process(getMessage(Buf, SourceMgr::DK_Warning, "unrelated"),
                       /*reportUnexpected=*/false),
      MatchResult::Ignored);
  EXPECT_EQ(
      Verifier.process(getMessage(Buf, SourceMgr::DK_Error, "expected msg"),
                       /*reportUnexpected=*/false),
      MatchResult::Matched);
  EXPECT_TRUE(Verifier.verify());
  EXPECT_EQ(Output, "");
}

TEST_F(SourceMgrDiagnosticVerifierTest, UnknownFileNeverMatches) {
  addBuffer("# expected-error {{anything}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  SourceMgrDiagnosticVerifier::Location Loc{/*buf=*/nullptr, /*lineNo=*/1,
                                            /*colNo=*/1};
  EXPECT_EQ(Verifier.process(SourceMgr::DK_Error, Loc, "anything"),
            MatchResult::Unexpected);
  EXPECT_FALSE(Verifier.verify());
}

TEST_F(SourceMgrDiagnosticVerifierTest, InstalledAsDiagHandler) {
  addBuffer("op\n# expected-warning@-1 {{bad op}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);
  SM.setDiagHandler(
      [](const SMDiagnostic &Diag, void *Ctx) {
        static_cast<SourceMgrDiagnosticVerifier *>(Ctx)->process(Diag);
      },
      &Verifier);

  // The near-miss message must be printed to OS rather than re-entering the
  // verifier through the SourceMgr's diagnostic handler.
  SM.PrintMessage(
      SMLoc::getFromPointer(SM.getMemoryBuffer(1)->getBufferStart()),
      SourceMgr::DK_Error, "bad op");
  EXPECT_NE(StringRef(Output).find(
                "error: 'error' diagnostic emitted when expecting a 'warning'"),
            StringRef::npos);
  EXPECT_FALSE(Verifier.verify());
}

TEST_F(SourceMgrDiagnosticVerifierTest, CustomPrefixesAndCommentScoping) {
  const MemoryBuffer *Buf =
      addBuffer("op\n"
                "// check-error@-1 {{first prefix}}\n"
                "# alt-warning@-2 {{second prefix}}\n"
                "// uncheck-error@-3 {{longer prefix ignored}}\n"
                "// other-check-error@-4 {{hyphenated prefix ignored}}\n"
                "// expected-error@-5 {{default prefix ignored}}\n"
                "op check-error {{outside comment ignored}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS, {"check", "alt"}, {"//", "#"});

  EXPECT_EQ(
      Verifier.process(getMessage(Buf, SourceMgr::DK_Error, "first prefix")),
      MatchResult::Matched);
  EXPECT_EQ(
      Verifier.process(getMessage(Buf, SourceMgr::DK_Warning, "second prefix")),
      MatchResult::Matched);
  EXPECT_TRUE(Verifier.verify());
  EXPECT_EQ(Output, "");
}

TEST_F(SourceMgrDiagnosticVerifierTest, DirectiveMatchesDuplicates) {
  const MemoryBuffer *Buf = addBuffer("op\n# expected-error@-1 {{dup}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  SMDiagnostic Dup = getMessage(Buf, SourceMgr::DK_Error, "dup");
  EXPECT_EQ(Verifier.process(Dup), MatchResult::Matched);
  EXPECT_EQ(Verifier.process(Dup), MatchResult::Matched);
  EXPECT_TRUE(Verifier.verify());
  EXPECT_EQ(Output, "");
}

TEST_F(SourceMgrDiagnosticVerifierTest, OneDiagPerDirective) {
  const MemoryBuffer *Buf = addBuffer("op\n"
                                      "# expected-error@-1 {{dup}}\n"
                                      "# expected-error@-2 {{dup}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS, {"expected"}, {},
                                       /*oneDiagPerDirective=*/true);

  // Each directive matches one diagnostic, so the third one is unexpected.
  SMDiagnostic Dup = getMessage(Buf, SourceMgr::DK_Error, "dup");
  EXPECT_EQ(Verifier.process(Dup), MatchResult::Matched);
  EXPECT_EQ(Verifier.process(Dup), MatchResult::Matched);
  EXPECT_EQ(Verifier.process(Dup), MatchResult::Unexpected);
  EXPECT_FALSE(Verifier.verify());
  EXPECT_EQ(StringRef(Output).count("error: dup"), 1u);
  EXPECT_EQ(StringRef(Output).find("was not produced"), StringRef::npos);
}

TEST_F(SourceMgrDiagnosticVerifierTest, ColumnChecking) {
  const MemoryBuffer *Buf = addBuffer("bad_op arg1, arg2\n"
                                      "# expected-error@-1:8 {{bad arg1}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  // Column 8 ('arg1') matches; column 1 ('bad_op') does not.
  SMLoc Col1Loc = SMLoc::getFromPointer(Buf->getBufferStart());
  SMLoc Col8Loc = SMLoc::getFromPointer(Buf->getBufferStart() + 7);
  EXPECT_EQ(
      Verifier.process(SM.GetMessage(Col1Loc, SourceMgr::DK_Error, "bad arg1")),
      MatchResult::Unexpected);
  EXPECT_EQ(
      Verifier.process(SM.GetMessage(Col8Loc, SourceMgr::DK_Error, "bad arg1")),
      MatchResult::Matched);
  EXPECT_FALSE(Verifier.verify());
}

TEST_F(SourceMgrDiagnosticVerifierTest, LineZeroAndLeadingZeroOffsets) {
  const MemoryBuffer *Buf =
      addBuffer("// expected-error@+02 {{decimal leading zero}}\n"
                "// expected-remark@-2 {{implicit module at line 0}}\n"
                "op\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  using Location = SourceMgrDiagnosticVerifier::Location;
  EXPECT_EQ(Verifier.process(SourceMgr::DK_Remark, Location{Buf, 0, 0},
                             "implicit module at line 0"),
            MatchResult::Matched);
  EXPECT_EQ(Verifier.process(SourceMgr::DK_Error, Location{Buf, 3, 1},
                             "decimal leading zero"),
            MatchResult::Matched);
  EXPECT_TRUE(Verifier.verify());
  EXPECT_EQ(Output, "");
}

TEST_F(SourceMgrDiagnosticVerifierTest, OutOfBoundsLineOffsets) {
  addBuffer("op\n"
            "# expected-error@-3 {{before start}}\n"
            "# expected-error@+5 {{past end}}\n"
            "# expected-error@+99999999999 {{overflow}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  EXPECT_FALSE(Verifier.verify());
  EXPECT_NE(StringRef(Output).find(
                "error: line offset '-3' is before the start of the file"),
            StringRef::npos);
  EXPECT_NE(StringRef(Output).find(
                "error: line offset '+5' is past the end of the file"),
            StringRef::npos);
  EXPECT_NE(StringRef(Output).find("error: invalid line offset '+99999999999'"),
            StringRef::npos);
}

TEST_F(SourceMgrDiagnosticVerifierTest, BuffersWithSameName) {
  const MemoryBuffer *Buf1 =
      addBuffer("op\n# expected-error@-1 {{first}}\n", "same.s");
  const MemoryBuffer *Buf2 =
      addBuffer("op\n# expected-error@-1 {{second}}\n", "same.s");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  // Directives only match diagnostics in their own buffer.
  EXPECT_EQ(Verifier.process(getMessage(Buf2, SourceMgr::DK_Error, "first")),
            MatchResult::Unexpected);
  EXPECT_EQ(Verifier.process(getMessage(Buf1, SourceMgr::DK_Error, "first")),
            MatchResult::Matched);
  EXPECT_EQ(Verifier.process(getMessage(Buf2, SourceMgr::DK_Error, "second")),
            MatchResult::Matched);
  EXPECT_FALSE(Verifier.verify());
}

TEST_F(SourceMgrDiagnosticVerifierTest, BufferAddedLater) {
  addBuffer("op\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);
  // E.g. an included file without any diagnostics.
  addBuffer("op\n# expected-error@-1 {{missing}}\n", "inc.s");

  EXPECT_FALSE(Verifier.verify());
  EXPECT_NE(StringRef(Output).find(
                "error: expected error \"missing\" was not produced"),
            StringRef::npos);

  // Calling verify() again does not report the same directives again.
  std::string FirstOutput = Output;
  EXPECT_FALSE(Verifier.verify());
  EXPECT_EQ(Output, FirstOutput);
}

TEST_F(SourceMgrDiagnosticVerifierTest, PreferUnmatchedDirective) {
  const MemoryBuffer *Buf =
      addBuffer("op\n"
                "# expected-error@-1 {{operand}}\n"
                "# expected-error@-2 {{invalid operand}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  // Both diagnostics match the first directive, but the second one must be
  // matched by the second directive.
  EXPECT_EQ(Verifier.process(getMessage(Buf, SourceMgr::DK_Error, "operand")),
            MatchResult::Matched);
  EXPECT_EQ(
      Verifier.process(getMessage(Buf, SourceMgr::DK_Error, "invalid operand")),
      MatchResult::Matched);
  EXPECT_TRUE(Verifier.verify());
  EXPECT_EQ(Output, "");
}

TEST_F(SourceMgrDiagnosticVerifierTest, AlreadyMatchedIsNotNearMiss) {
  const MemoryBuffer *Buf = addBuffer("op\n# expected-error@-1 {{operand}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  EXPECT_EQ(
      Verifier.process(getMessage(Buf, SourceMgr::DK_Error, "bad operand")),
      MatchResult::Matched);
  EXPECT_EQ(
      Verifier.process(getMessage(Buf, SourceMgr::DK_Note, "note operand")),
      MatchResult::Unexpected);
  EXPECT_FALSE(Verifier.verify());
  EXPECT_EQ(StringRef(Output).find("diagnostic emitted when expecting"),
            StringRef::npos);
}

TEST_F(SourceMgrDiagnosticVerifierTest, ErrorHighlightsWholeDirective) {
  addBuffer("op\n# expected-error@-1 {{missing}}\n");
  SourceMgrDiagnosticVerifier Verifier(SM, OS);

  EXPECT_FALSE(Verifier.verify());
  EXPECT_NE(StringRef(Output).find("# expected-error@-1 {{missing}}\n"
                                   "  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"),
            StringRef::npos)
      << Output;
}
