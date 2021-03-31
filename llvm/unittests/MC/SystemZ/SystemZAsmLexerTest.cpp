//===- llvm/unittests/MC/SystemZ/SystemZAsmLexerTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Come up with our hacked version of MCAsmInfo.
// This hacked version derives from the main MCAsmInfo instance.
// Here, we're free to override whatever we want, without polluting
// the main MCAsmInfo interface.
class MockedUpMCAsmInfo : public MCAsmInfo {
public:
  void setRestrictCommentStringToStartOfStatement(bool Value) {
    RestrictCommentStringToStartOfStatement = Value;
  }
  void setCommentString(StringRef Value) { CommentString = Value; }
};

// Setup a testing class that the GTest framework can call.
class SystemZAsmLexerTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    LLVMInitializeSystemZTargetInfo();
    LLVMInitializeSystemZTargetMC();
  }

  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MockedUpMCAsmInfo> MUPMAI;
  std::unique_ptr<const MCInstrInfo> MII;
  std::unique_ptr<MCStreamer> Str;
  std::unique_ptr<MCAsmParser> Parser;
  std::unique_ptr<MCContext> Ctx;

  SourceMgr SrcMgr;
  std::string TripleName;
  llvm::Triple Triple;
  const Target *TheTarget;

  const MCTargetOptions MCOptions;
  MCObjectFileInfo MOFI;

  SystemZAsmLexerTest() {
    // We will use the SystemZ triple, because of missing
    // Object File and Streamer support for the z/OS target.
    TripleName = "s390x-ibm-linux";
    Triple = llvm::Triple(TripleName);

    std::string Error;
    TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
    EXPECT_NE(TheTarget, nullptr);

    MRI.reset(TheTarget->createMCRegInfo(TripleName));
    EXPECT_NE(MRI, nullptr);

    std::unique_ptr<MCAsmInfo> MAI;
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
    EXPECT_NE(MAI, nullptr);

    // Now we cast to our mocked up version of MCAsmInfo.
    MUPMAI.reset(static_cast<MockedUpMCAsmInfo *>(MAI.release()));
    // MUPMAI should "hold" MAI.
    EXPECT_NE(MUPMAI, nullptr);
    // After releasing, MAI should now be null.
    EXPECT_EQ(MAI, nullptr);
  }

  void setupCallToAsmParser(StringRef AsmStr) {
    std::unique_ptr<MemoryBuffer> Buffer(MemoryBuffer::getMemBuffer(AsmStr));
    SrcMgr.AddNewSourceBuffer(std::move(Buffer), SMLoc());
    EXPECT_EQ(Buffer, nullptr);

    Ctx.reset(
        new MCContext(MUPMAI.get(), MRI.get(), &MOFI, &SrcMgr, &MCOptions));
    MOFI.InitMCObjectFileInfo(Triple, false, *Ctx, false);

    Str.reset(TheTarget->createNullStreamer(*Ctx));

    Parser.reset(createMCAsmParser(SrcMgr, *Ctx, *Str, *MUPMAI));
    // Lex initially to get the string.
    Parser->getLexer().Lex();
  }

  void lexAndCheckTokens(StringRef AsmStr,
                         SmallVector<AsmToken::TokenKind> ExpectedTokens) {
    // Get reference to AsmLexer.
    MCAsmLexer &Lexer = Parser->getLexer();
    // Loop through all expected tokens checking one by one.
    for (size_t I = 0; I < ExpectedTokens.size(); ++I) {
      EXPECT_EQ(Lexer.getTok().getKind(), ExpectedTokens[I]);
      Lexer.Lex();
    }
  }
};

TEST_F(SystemZAsmLexerTest, CheckDontRestrictCommentStringToStartOfStatement) {
  StringRef AsmStr = "jne #-4";

  // Setup.
  setupCallToAsmParser(AsmStr);

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement});
  lexAndCheckTokens(AsmStr /* "jne #-4" */, ExpectedTokens);
}

// Testing MCAsmInfo's RestrictCommentStringToStartOfStatement attribute.
TEST_F(SystemZAsmLexerTest, CheckRestrictCommentStringToStartOfStatement) {
  StringRef AsmStr = "jne #-4";

  // Setup.
  MUPMAI->setRestrictCommentStringToStartOfStatement(true);
  setupCallToAsmParser(AsmStr);

  // When we are restricting the comment string to only the start of the
  // statement, The sequence of tokens we are expecting are: Identifier - "jne"
  // Hash - '#'
  // Minus - '-'
  // Integer - '4'
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::Hash, AsmToken::Minus,
       AsmToken::Integer});
  lexAndCheckTokens(AsmStr /* "jne #-4" */, ExpectedTokens);
}

// Test HLASM Comment Syntax ('*')
TEST_F(SystemZAsmLexerTest, CheckHLASMComment) {
  StringRef AsmStr = "* lhi 1,10";

  // Setup.
  MUPMAI->setCommentString("*");
  setupCallToAsmParser(AsmStr);

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr /* "* lhi 1,10" */, ExpectedTokens);
}
} // end anonymous namespace
