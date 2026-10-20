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

  SourceMgrDiagnosticVerifier::Location Loc{/*buf=*/nullptr, /*lineNo=*/1};
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
