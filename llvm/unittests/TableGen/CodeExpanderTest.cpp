//===- llvm/unittest/TableGen/CodeExpanderTest.cpp - Tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalISel/CodeExpander.h"
#include "GlobalISel/CodeExpansions.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

static StringRef bufferize(StringRef Str) {
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBufferCopy(Str, "TestBuffer");
  StringRef StrBufferRef = Buffer->getBuffer();
  SrcMgr.AddNewSourceBuffer(std::move(Buffer), SMLoc());
  return StrBufferRef;
}

class RAIIDiagnosticChecker {
  std::string EmittedDiags;
  raw_string_ostream OS;
  std::vector<SMDiagnostic> Expected;
  std::vector<SMDiagnostic> Received;

public:
  RAIIDiagnosticChecker() : OS(EmittedDiags) {
    SrcMgr.setDiagHandler(handler, this);
  }
  ~RAIIDiagnosticChecker() {
    SrcMgr.setDiagHandler(nullptr);
    EXPECT_EQ(Received.size(), Expected.size());
    for (unsigned i = 0; i < Received.size() && i < Expected.size(); ++i) {
      EXPECT_EQ(Received[i].getLoc(), Expected[i].getLoc());
      EXPECT_EQ(Received[i].getFilename(), Expected[i].getFilename());
      EXPECT_EQ(Received[i].getKind(), Expected[i].getKind());
      EXPECT_EQ(Received[i].getLineNo(), Expected[i].getLineNo());
      EXPECT_EQ(Received[i].getColumnNo(), Expected[i].getColumnNo());
      EXPECT_EQ(Received[i].getMessage(), Expected[i].getMessage());
      EXPECT_EQ(Received[i].getLineContents(), Expected[i].getLineContents());
      EXPECT_EQ(Received[i].getRanges(), Expected[i].getRanges());
    }

    if (testing::Test::HasFailure())
      errs() << "Emitted diagnostic:\n" << OS.str();
  }

  void expect(SMDiagnostic D) { Expected.push_back(D); }

  void diag(const SMDiagnostic &D) {
    Received.push_back(D);
  }

  static void handler(const SMDiagnostic &D, void *Context) {
    RAIIDiagnosticChecker *Self = static_cast<RAIIDiagnosticChecker *>(Context);
    Self->diag(D);
    SrcMgr.setDiagHandler(nullptr);
    SrcMgr.PrintMessage(Self->OS, D);
    SrcMgr.setDiagHandler(handler, Context);
  };
};

TEST(CodeExpander, NoExpansions) {
  std::string Result;
  raw_string_ostream OS(Result);
  CodeExpansions Expansions;

  RAIIDiagnosticChecker DiagChecker;
  CodeExpander("No expansions", Expansions, SMLoc(), false).emit(OS);
  EXPECT_EQ(OS.str(), "No expansions");
}

// Indentation is applied to all lines except the first
TEST(CodeExpander, Indentation) {
  std::string Result;
  raw_string_ostream OS(Result);
  CodeExpansions Expansions;

  RAIIDiagnosticChecker DiagChecker;
  CodeExpander("No expansions\nsecond line\nthird line", Expansions, SMLoc(),
               false, "  ")
      .emit(OS);
  EXPECT_EQ(OS.str(), "No expansions\n  second line\n  third line");
}

// \ is an escape character that removes special meanings from the next
// character.
TEST(CodeExpander, Escape) {
  std::string Result;
  raw_string_ostream OS(Result);
  CodeExpansions Expansions;

  RAIIDiagnosticChecker DiagChecker;
  CodeExpander("\\\\\\a\\$", Expansions, SMLoc(), false).emit(OS);
  EXPECT_EQ(OS.str(), "\\a$");
}

// $foo is not an expansion. It should warn though.
TEST(CodeExpander, NotAnExpansion) {
  std::string Result;
  raw_string_ostream OS(Result);
  CodeExpansions Expansions;

  RAIIDiagnosticChecker DiagChecker;
  StringRef In = bufferize(" $foo");
  CodeExpander(" $foo", Expansions, SMLoc::getFromPointer(In.data()), false)
      .emit(OS);
  EXPECT_EQ(OS.str(), " $foo");
  DiagChecker.expect(SMDiagnostic(
      SrcMgr, SMLoc::getFromPointer(In.data()), "TestBuffer", 1, 0,
      SourceMgr::DK_Warning, "Assuming missing escape character: \\$", " $foo", {}));
}

// \$foo is not an expansion but shouldn't warn as it's using the escape.
TEST(CodeExpander, EscapedNotAnExpansion) {
  std::string Result;
  raw_string_ostream OS(Result);
  CodeExpansions Expansions;

  RAIIDiagnosticChecker DiagChecker;
  CodeExpander("\\$foo", Expansions, SMLoc(), false).emit(OS);
  EXPECT_EQ(OS.str(), "$foo");
}

// \${foo is not an expansion but shouldn't warn as it's using the escape.
TEST(CodeExpander, EscapedUnterminatedExpansion) {
  std::string Result;
  raw_string_ostream OS(Result);
  CodeExpansions Expansions;

  RAIIDiagnosticChecker DiagChecker;
  CodeExpander("\\${foo", Expansions, SMLoc(), false).emit(OS);
  EXPECT_EQ(OS.str(), "${foo");
}

// \${foo is not an expansion but shouldn't warn as it's using the escape.
TEST(CodeExpander, EscapedExpansion) {
  std::string Result;
  raw_string_ostream OS(Result);
  CodeExpansions Expansions;

  RAIIDiagnosticChecker DiagChecker;
  CodeExpander("\\${foo}", Expansions, SMLoc(), false).emit(OS);
  EXPECT_EQ(OS.str(), "${foo}");
}

// ${foo} is an undefined expansion and should error.
TEST(CodeExpander, UndefinedExpansion) {
  std::string Result;
  raw_string_ostream OS(Result);
  CodeExpansions Expansions;
  Expansions.declare("bar", "expansion");

  RAIIDiagnosticChecker DiagChecker;
  CodeExpander("${foo}${bar}", Expansions, SMLoc(), false).emit(OS);
  EXPECT_EQ(OS.str(), "expansion");
  DiagChecker.expect(
      SMDiagnostic(SrcMgr, SMLoc(), "<unknown>", 0, -1, SourceMgr::DK_Error,
                   "Attempt to expand an undeclared variable 'foo'", "", {}));
}

// ${bar is an unterminated expansion. Warn and implicitly terminate it.
TEST(CodeExpander, UnterminatedExpansion) {
  std::string Result;
  raw_string_ostream OS(Result);
  CodeExpansions Expansions;
  Expansions.declare("bar", "expansion");

  RAIIDiagnosticChecker DiagChecker;
  StringRef In = bufferize(" ${bar");
  CodeExpander(In, Expansions, SMLoc::getFromPointer(In.data()), false)
      .emit(OS);
  EXPECT_EQ(OS.str(), " expansion");
  DiagChecker.expect(SMDiagnostic(SrcMgr, SMLoc::getFromPointer(In.data()),
                                  "TestBuffer", 1, 0, SourceMgr::DK_Warning,
                                  "Unterminated expansion '${bar'", " ${bar", {}));
}
