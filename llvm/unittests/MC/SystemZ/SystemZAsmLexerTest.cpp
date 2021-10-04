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

// Setup a testing class that the GTest framework can call.
class SystemZAsmLexerTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    LLVMInitializeSystemZTargetInfo();
    LLVMInitializeSystemZTargetMC();
    LLVMInitializeSystemZAsmParser();
  }

  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<const MCInstrInfo> MII;
  std::unique_ptr<MCObjectFileInfo> MOFI;
  std::unique_ptr<MCStreamer> Str;
  std::unique_ptr<MCAsmParser> Parser;
  std::unique_ptr<MCContext> Ctx;
  std::unique_ptr<MCSubtargetInfo> STI;
  std::unique_ptr<MCTargetAsmParser> TargetAsmParser;

  SourceMgr SrcMgr;
  std::string TripleName;
  llvm::Triple Triple;
  const Target *TheTarget;

  const MCTargetOptions MCOptions;

  SystemZAsmLexerTest() = delete;

  SystemZAsmLexerTest(std::string SystemZTriple) {
    // We will use the SystemZ triple, because of missing
    // Object File and Streamer support for the z/OS target.
    TripleName = SystemZTriple;
    Triple = llvm::Triple(TripleName);

    std::string Error;
    TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
    EXPECT_NE(TheTarget, nullptr);

    MRI.reset(TheTarget->createMCRegInfo(TripleName));
    EXPECT_NE(MRI, nullptr);

    MII.reset(TheTarget->createMCInstrInfo());
    EXPECT_NE(MII, nullptr);

    STI.reset(TheTarget->createMCSubtargetInfo(TripleName, "z10", ""));
    EXPECT_NE(STI, nullptr);

    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
    EXPECT_NE(MAI, nullptr);
  }

  void setupCallToAsmParser(StringRef AsmStr) {
    std::unique_ptr<MemoryBuffer> Buffer(MemoryBuffer::getMemBuffer(AsmStr));
    SrcMgr.AddNewSourceBuffer(std::move(Buffer), SMLoc());
    EXPECT_EQ(Buffer, nullptr);

    Ctx.reset(new MCContext(Triple, MAI.get(), MRI.get(), STI.get(), &SrcMgr,
                            &MCOptions));
    MOFI.reset(TheTarget->createMCObjectFileInfo(*Ctx, /*PIC=*/false,
                                                 /*LargeCodeModel=*/false));
    Ctx->setObjectFileInfo(MOFI.get());

    Str.reset(TheTarget->createNullStreamer(*Ctx));

    Parser.reset(createMCAsmParser(SrcMgr, *Ctx, *Str, *MAI));

    TargetAsmParser.reset(
        TheTarget->createMCAsmParser(*STI, *Parser, *MII, MCOptions));
    Parser->setTargetParser(*TargetAsmParser);
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

  void lexAndCheckIntegerTokensAndValues(StringRef AsmStr,
                                         SmallVector<int64_t> ExpectedValues) {
    // Get reference to AsmLexer.
    MCAsmLexer &Lexer = Parser->getLexer();
    // Loop through all expected tokens and expected values.
    for (size_t I = 0; I < ExpectedValues.size(); ++I) {
      // Skip any EndOfStatement tokens, we're not concerned with them.
      if (Lexer.getTok().getKind() == AsmToken::EndOfStatement)
        continue;
      EXPECT_EQ(Lexer.getTok().getKind(), AsmToken::Integer);
      EXPECT_EQ(Lexer.getTok().getIntVal(), ExpectedValues[I]);
      Lexer.Lex();
    }
  }
};

class SystemZAsmLexerLinux : public SystemZAsmLexerTest {
protected:
  SystemZAsmLexerLinux() : SystemZAsmLexerTest("s390x-ibm-linux") {}
};

class SystemZAsmLexerZOS : public SystemZAsmLexerTest {
protected:
  SystemZAsmLexerZOS() : SystemZAsmLexerTest("s390x-ibm-zos") {}
};

TEST_F(SystemZAsmLexerLinux, CheckDontRestrictCommentStringToStartOfStatement) {
  StringRef AsmStr = "jne #-4";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement});
  lexAndCheckTokens(AsmStr /* "jne #-4" */, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckRestrictCommentStringToStartOfStatement) {
  StringRef AsmStr = "jne #-4";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // When we are restricting the comment string to only the start of the
  // statement, The sequence of tokens we are expecting are: Identifier - "jne"
  // Hash - '#'
  // Minus - '-'
  // Integer - '4'
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::Space, AsmToken::Identifier});
  lexAndCheckTokens(AsmStr /* "jne #-4" */, ExpectedTokens);
}

// Test HLASM Comment Syntax ('*')
TEST_F(SystemZAsmLexerZOS, CheckHLASMComment) {
  StringRef AsmStr = "* lhi 1,10";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr /* "* lhi 1,10" */, ExpectedTokens);
}

TEST_F(SystemZAsmLexerLinux, CheckHashDefault) {
  StringRef AsmStr = "lh#123";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // "lh" -> Identifier
  // "#123" -> EndOfStatement (Lexed as a comment since CommentString is "#")
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

// Test if "#" is accepted as an Identifier
TEST_F(SystemZAsmLexerZOS, CheckAllowHashInIdentifier) {
  StringRef AsmStr = "lh#123";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // "lh123" -> Identifier
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckAllowHashInIdentifier2) {
  StringRef AsmStr = "lh#12*3";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // "lh#12" -> Identifier
  // "*" -> Star
  // "3" -> Integer
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::Star, AsmToken::Integer,
       AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerLinux, DontCheckStrictCommentString) {
  StringRef AsmStr = "# abc\n/* def *///  xyz";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::EndOfStatement, AsmToken::Comment, AsmToken::EndOfStatement,
       AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckStrictCommentString) {
  StringRef AsmStr = "# abc\n/* def *///  xyz";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(AsmToken::Identifier);     // "#"
  ExpectedTokens.push_back(AsmToken::Space);          // " "
  ExpectedTokens.push_back(AsmToken::Identifier);     // "abc"
  ExpectedTokens.push_back(AsmToken::EndOfStatement); // "\n"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Star);           // "*"
  ExpectedTokens.push_back(AsmToken::Space);          // " "
  ExpectedTokens.push_back(AsmToken::Identifier);     // "def"
  ExpectedTokens.push_back(AsmToken::Space);          // " "
  ExpectedTokens.push_back(AsmToken::Star);           // "*"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Space);          // " "
  ExpectedTokens.push_back(AsmToken::Identifier);     // "xyz"
  ExpectedTokens.push_back(AsmToken::EndOfStatement);
  ExpectedTokens.push_back(AsmToken::Eof);

  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckValidHLASMIntegers) {
  StringRef AsmStr = "123\n000123\n1999\n007\n12300\n12021\n";
  // StringRef AsmStr = "123";
  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // SmallVector<int64_t> ExpectedValues({123});
  SmallVector<int64_t> ExpectedValues({123, 123, 1999, 7, 12300, 12021});
  lexAndCheckIntegerTokensAndValues(AsmStr, ExpectedValues);
}

TEST_F(SystemZAsmLexerZOS, CheckInvalidHLASMIntegers) {
  StringRef AsmStr = "0b0101\n0xDEADBEEF\nfffh\n.133\n";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(AsmToken::Integer);        // "0"
  ExpectedTokens.push_back(AsmToken::Identifier);     // "b0101"
  ExpectedTokens.push_back(AsmToken::EndOfStatement); // "\n"
  ExpectedTokens.push_back(AsmToken::Integer);        // "0"
  ExpectedTokens.push_back(AsmToken::Identifier);     // "xDEADBEEF"
  ExpectedTokens.push_back(AsmToken::EndOfStatement); // "\n"
  ExpectedTokens.push_back(AsmToken::Identifier);     // "fffh"
  ExpectedTokens.push_back(AsmToken::EndOfStatement); // "\n"
  ExpectedTokens.push_back(AsmToken::Real);           // ".133"
  ExpectedTokens.push_back(AsmToken::EndOfStatement); // "\n"
  ExpectedTokens.push_back(AsmToken::Eof);
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerLinux, CheckDefaultIntegers) {
  StringRef AsmStr = "0b0101\n0xDEADBEEF\nfffh\n";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<int64_t> ExpectedValues({5, 0xDEADBEEF, 0xFFF});
  lexAndCheckIntegerTokensAndValues(AsmStr, ExpectedValues);
}

TEST_F(SystemZAsmLexerLinux, CheckDefaultFloats) {
  StringRef AsmStr = "0.333\n1.3\n2.5\n3.0\n";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens;

  for (int I = 0; I < 4; ++I)
    ExpectedTokens.insert(ExpectedTokens.begin(),
                          {AsmToken::Real, AsmToken::EndOfStatement});

  ExpectedTokens.push_back(AsmToken::Eof);
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerLinux, CheckDefaultQuestionAtStartOfIdentifier) {
  StringRef AsmStr = "?lh1?23";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Error, AsmToken::Identifier, AsmToken::EndOfStatement,
       AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerLinux, CheckDefaultAtAtStartOfIdentifier) {
  StringRef AsmStr = "@@lh1?23";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::At, AsmToken::At, AsmToken::Identifier,
       AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckAcceptAtAtStartOfIdentifier) {
  StringRef AsmStr = "@@lh1?23";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerLinux, CheckDefaultDollarAtStartOfIdentifier) {
  StringRef AsmStr = "$$ac$c";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Dollar, AsmToken::Dollar, AsmToken::Identifier,
       AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckAcceptDollarAtStartOfIdentifier) {
  StringRef AsmStr = "$$ab$c";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckAcceptHashAtStartOfIdentifier) {
  StringRef AsmStr = "##a#b$c";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerLinux, CheckAcceptHashAtStartOfIdentifier2) {
  StringRef AsmStr = "##a#b$c";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // By default, the CommentString attribute is set to "#".
  // Hence, "##a#b$c" is lexed as a line comment irrespective
  // of whether the AllowHashAtStartOfIdentifier attribute is set to true.
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckAcceptHashAtStartOfIdentifier3) {
  StringRef AsmStr = "##a#b$c";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckAcceptHashAtStartOfIdentifier4) {
  StringRef AsmStr = "##a#b$c";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // Since, the AllowAdditionalComments attribute is set to false,
  // only strings starting with the CommentString attribute are
  // lexed as possible comments.
  // Hence, "##a$b$c" is lexed as an Identifier because the
  // AllowHashAtStartOfIdentifier attribute is set to true.
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckRejectDotAsCurrentPC) {
  StringRef AsmStr = ".-4";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  const MCExpr *Expr;
  bool ParsePrimaryExpr = Parser->parseExpression(Expr);
  EXPECT_EQ(ParsePrimaryExpr, true);
  EXPECT_EQ(Parser->hasPendingError(), true);
}

TEST_F(SystemZAsmLexerLinux, CheckRejectStarAsCurrentPC) {
  StringRef AsmStr = "*-4";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  const MCExpr *Expr;
  bool ParsePrimaryExpr = Parser->parseExpression(Expr);
  EXPECT_EQ(ParsePrimaryExpr, true);
  EXPECT_EQ(Parser->hasPendingError(), true);
}

TEST_F(SystemZAsmLexerZOS, CheckRejectCharLiterals) {
  StringRef AsmStr = "abc 'd'";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::Space, AsmToken::Error, AsmToken::Error,
       AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckRejectStringLiterals) {
  StringRef AsmStr = "abc \"ef\"";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::Space, AsmToken::Error,
       AsmToken::Identifier, AsmToken::Error, AsmToken::EndOfStatement,
       AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerZOS, CheckPrintAcceptableSymbol) {
  std::string AsmStr = "ab13_$.@";
  EXPECT_EQ(true, MAI->isValidUnquotedName(AsmStr));
  AsmStr += "#";
  EXPECT_EQ(true, MAI->isValidUnquotedName(AsmStr));
}

TEST_F(SystemZAsmLexerLinux, CheckPrintAcceptableSymbol) {
  std::string AsmStr = "ab13_$.@";
  EXPECT_EQ(true, MAI->isValidUnquotedName(AsmStr));
  AsmStr += "#";
  EXPECT_EQ(false, MAI->isValidUnquotedName(AsmStr));
}

TEST_F(SystemZAsmLexerZOS, CheckLabelCaseUpperCase) {
  StringRef AsmStr = "label";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  const MCExpr *Expr;
  bool ParsePrimaryExpr = Parser->parseExpression(Expr);
  EXPECT_EQ(ParsePrimaryExpr, false);

  const MCSymbolRefExpr *SymbolExpr = dyn_cast<MCSymbolRefExpr>(Expr);
  EXPECT_NE(SymbolExpr, nullptr);
  EXPECT_NE(&SymbolExpr->getSymbol(), nullptr);
  EXPECT_EQ((&SymbolExpr->getSymbol())->getName(), StringRef("LABEL"));
}

TEST_F(SystemZAsmLexerLinux, CheckLabelUpperCase2) {
  StringRef AsmStr = "label";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  const MCExpr *Expr;
  bool ParsePrimaryExpr = Parser->parseExpression(Expr);
  EXPECT_EQ(ParsePrimaryExpr, false);

  const MCSymbolRefExpr *SymbolExpr = dyn_cast<MCSymbolRefExpr>(Expr);
  EXPECT_NE(SymbolExpr, nullptr);
  EXPECT_NE(&SymbolExpr->getSymbol(), nullptr);
  EXPECT_EQ((&SymbolExpr->getSymbol())->getName(), StringRef("label"));
}
} // end anonymous namespace
