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
  void setAllowAdditionalComments(bool Value) {
    AllowAdditionalComments = Value;
  }
  void setAllowQuestionAtStartOfIdentifier(bool Value) {
    AllowQuestionAtStartOfIdentifier = Value;
  }
  void setAllowAtAtStartOfIdentifier(bool Value) {
    AllowAtAtStartOfIdentifier = Value;
  }
  void setAllowDollarAtStartOfIdentifier(bool Value) {
    AllowDollarAtStartOfIdentifier = Value;
  }
  void setAllowHashAtStartOfIdentifier(bool Value) {
    AllowHashAtStartOfIdentifier = Value;
  }
  void setAllowDotIsPC(bool Value) { DotIsPC = Value; }
  void setAssemblerDialect(unsigned Value) { AssemblerDialect = Value; }
  void setEmitLabelsInUpperCase(bool Value) { EmitLabelsInUpperCase = Value; }
};

// Setup a testing class that the GTest framework can call.
class SystemZAsmLexerTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    LLVMInitializeSystemZTargetInfo();
    LLVMInitializeSystemZTargetMC();
    LLVMInitializeSystemZAsmParser();
  }

  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MockedUpMCAsmInfo> MUPMAI;
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

    MII.reset(TheTarget->createMCInstrInfo());
    EXPECT_NE(MII, nullptr);

    STI.reset(TheTarget->createMCSubtargetInfo(TripleName, "z10", ""));
    EXPECT_NE(STI, nullptr);

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

    Ctx.reset(new MCContext(Triple, MUPMAI.get(), MRI.get(), STI.get(), &SrcMgr,
                            &MCOptions));
    MOFI.reset(TheTarget->createMCObjectFileInfo(*Ctx, /*PIC=*/false,
                                                 /*LargeCodeModel=*/false));
    Ctx->setObjectFileInfo(MOFI.get());

    Str.reset(TheTarget->createNullStreamer(*Ctx));

    Parser.reset(createMCAsmParser(SrcMgr, *Ctx, *Str, *MUPMAI));

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

TEST_F(SystemZAsmLexerTest, CheckDontRestrictCommentStringToStartOfStatement) {
  StringRef AsmStr = "jne #-4";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

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

  // Lex initially to get the string.
  Parser->getLexer().Lex();

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

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr /* "* lhi 1,10" */, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckHashDefault) {
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
TEST_F(SystemZAsmLexerTest, CheckAllowHashInIdentifier) {
  StringRef AsmStr = "lh#123";

  // Setup.
  setupCallToAsmParser(AsmStr);
  Parser->getLexer().setAllowHashInIdentifier(true);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // "lh123" -> Identifier
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckAllowHashInIdentifier2) {
  StringRef AsmStr = "lh#12*3";

  // Setup.
  MUPMAI->setCommentString("*");
  MUPMAI->setRestrictCommentStringToStartOfStatement(true);
  setupCallToAsmParser(AsmStr);
  Parser->getLexer().setAllowHashInIdentifier(true);

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

TEST_F(SystemZAsmLexerTest, DontCheckStrictCommentString) {
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

TEST_F(SystemZAsmLexerTest, DontCheckStrictCommentString2) {
  StringRef AsmStr = "# abc\n/* def *///  xyz\n* rst";

  // Setup.
  MUPMAI->setCommentString("*");
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::EndOfStatement, AsmToken::Comment, AsmToken::EndOfStatement,
       AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckStrictCommentString) {
  StringRef AsmStr = "# abc\n/* def *///  xyz";

  // Setup.
  MUPMAI->setAllowAdditionalComments(false);
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // "# abc" -> still treated as a comment, since CommentString
  //            is set to "#"
  SmallVector<AsmToken::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(AsmToken::EndOfStatement); // "# abc\n"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Star);           // "*"
  ExpectedTokens.push_back(AsmToken::Identifier);     // "def"
  ExpectedTokens.push_back(AsmToken::Star);           // "*"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Identifier);     // "xyz"
  ExpectedTokens.push_back(AsmToken::EndOfStatement);
  ExpectedTokens.push_back(AsmToken::Eof);

  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckStrictCommentString2) {
  StringRef AsmStr = "// abc";

  // Setup.
  MUPMAI->setAllowAdditionalComments(false);
  MUPMAI->setCommentString("//");
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // "// abc" -> will still be treated as a comment because "//" is the
  //             CommentString
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr /* "// abc" */, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckStrictCommentString3) {
  StringRef AsmStr = "/* abc */";

  // Setup.
  MUPMAI->setAllowAdditionalComments(false);
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(AsmToken::Slash);
  ExpectedTokens.push_back(AsmToken::Star);
  ExpectedTokens.push_back(AsmToken::Identifier);
  ExpectedTokens.push_back(AsmToken::Star);
  ExpectedTokens.push_back(AsmToken::Slash);
  ExpectedTokens.push_back(AsmToken::EndOfStatement);
  ExpectedTokens.push_back(AsmToken::Eof);

  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckStrictCommentString4) {
  StringRef AsmStr = "# abc\n/* def *///  xyz";

  // Setup.
  MUPMAI->setCommentString("*");
  MUPMAI->setAllowAdditionalComments(false);
  MUPMAI->setRestrictCommentStringToStartOfStatement(true);
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(AsmToken::Hash);           // "#"
  ExpectedTokens.push_back(AsmToken::Identifier);     // "abc"
  ExpectedTokens.push_back(AsmToken::EndOfStatement); // "\n"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Star);           // "*"
  ExpectedTokens.push_back(AsmToken::Identifier);     // "def"
  ExpectedTokens.push_back(AsmToken::Star);           // "*"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::Identifier);     // "xyz"
  ExpectedTokens.push_back(AsmToken::EndOfStatement);
  ExpectedTokens.push_back(AsmToken::Eof);

  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckStrictCommentString5) {
  StringRef AsmStr = "#abc\n/* def */// xyz";

  // Setup.
  MUPMAI->setCommentString("*");
  MUPMAI->setAllowAdditionalComments(false);
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(AsmToken::Hash);           // "#"
  ExpectedTokens.push_back(AsmToken::Identifier);     // "abc"
  ExpectedTokens.push_back(AsmToken::EndOfStatement); // "\n"
  ExpectedTokens.push_back(AsmToken::Slash);          // "/"
  ExpectedTokens.push_back(AsmToken::EndOfStatement); // "* def */// xyz"
  ExpectedTokens.push_back(AsmToken::Eof);

  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckValidHLASMIntegers) {
  StringRef AsmStr = "123\n000123\n1999\n007\n12300\n12021\n";
  // StringRef AsmStr = "123";
  // Setup.
  setupCallToAsmParser(AsmStr);
  Parser->getLexer().setLexHLASMIntegers(true);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // SmallVector<int64_t> ExpectedValues({123});
  SmallVector<int64_t> ExpectedValues({123, 123, 1999, 7, 12300, 12021});
  lexAndCheckIntegerTokensAndValues(AsmStr, ExpectedValues);
}

TEST_F(SystemZAsmLexerTest, CheckInvalidHLASMIntegers) {
  StringRef AsmStr = "0b0101\n0xDEADBEEF\nfffh\n.133\n";

  // Setup.
  setupCallToAsmParser(AsmStr);
  Parser->getLexer().setLexHLASMIntegers(true);

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

TEST_F(SystemZAsmLexerTest, CheckDefaultIntegers) {
  StringRef AsmStr = "0b0101\n0xDEADBEEF\nfffh\n";

  // Setup.
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<int64_t> ExpectedValues({5, 0xDEADBEEF, 0xFFF});
  lexAndCheckIntegerTokensAndValues(AsmStr, ExpectedValues);
}

TEST_F(SystemZAsmLexerTest, CheckDefaultFloats) {
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

TEST_F(SystemZAsmLexerTest, CheckDefaultQuestionAtStartOfIdentifier) {
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

TEST_F(SystemZAsmLexerTest, CheckAcceptQuestionAtStartOfIdentifier) {
  StringRef AsmStr = "?????lh1?23";

  // Setup.
  MUPMAI->setAllowQuestionAtStartOfIdentifier(true);
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckDefaultAtAtStartOfIdentifier) {
  StringRef AsmStr = "@@lh1?23";

  // Setup.
  MUPMAI->setAllowQuestionAtStartOfIdentifier(true);
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::At, AsmToken::At, AsmToken::Identifier,
       AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckAcceptAtAtStartOfIdentifier) {
  StringRef AsmStr = "@@lh1?23";

  // Setup.
  MUPMAI->setAllowAtAtStartOfIdentifier(true);
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckAccpetAtAtStartOfIdentifier2) {
  StringRef AsmStr = "@@lj1?23";

  // Setup.
  MUPMAI->setCommentString("@");
  MUPMAI->setAllowAtAtStartOfIdentifier(true);
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // "@@lj1?23" -> still lexed as a comment as that takes precedence.
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckDefaultDollarAtStartOfIdentifier) {
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

TEST_F(SystemZAsmLexerTest, CheckAcceptDollarAtStartOfIdentifier) {
  StringRef AsmStr = "$$ab$c";

  // Setup.
  MUPMAI->setAllowDollarAtStartOfIdentifier(true);
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckAcceptHashAtStartOfIdentifier) {
  StringRef AsmStr = "##a#b$c";

  // Setup.
  MUPMAI->setAllowHashAtStartOfIdentifier(true);
  MUPMAI->setCommentString("*");
  MUPMAI->setAllowAdditionalComments(false);
  setupCallToAsmParser(AsmStr);
  Parser->getLexer().setAllowHashInIdentifier(true);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckAcceptHashAtStartOfIdentifier2) {
  StringRef AsmStr = "##a#b$c";

  // Setup.
  MUPMAI->setAllowHashAtStartOfIdentifier(true);
  setupCallToAsmParser(AsmStr);
  Parser->getLexer().setAllowHashInIdentifier(true);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // By default, the CommentString attribute is set to "#".
  // Hence, "##a#b$c" is lexed as a line comment irrespective
  // of whether the AllowHashAtStartOfIdentifier attribute is set to true.
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckAcceptHashAtStartOfIdentifier3) {
  StringRef AsmStr = "##a#b$c";

  // Setup.
  MUPMAI->setAllowHashAtStartOfIdentifier(true);
  MUPMAI->setCommentString("*");
  setupCallToAsmParser(AsmStr);
  Parser->getLexer().setAllowHashInIdentifier(true);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  // By default, the AsmLexer treats strings that start with "#"
  // as a line comment.
  // Hence, "##a$b$c" is lexed as a line comment irrespective
  // of whether the AllowHashAtStartOfIdentifier attribute is set to true.
  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckAcceptHashAtStartOfIdentifier4) {
  StringRef AsmStr = "##a#b$c";

  // Setup.
  MUPMAI->setAllowHashAtStartOfIdentifier(true);
  MUPMAI->setCommentString("*");
  MUPMAI->setAllowAdditionalComments(false);
  setupCallToAsmParser(AsmStr);
  Parser->getLexer().setAllowHashInIdentifier(true);

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

TEST_F(SystemZAsmLexerTest, CheckRejectDotAsCurrentPC) {
  StringRef AsmStr = ".-4";

  // Setup.
  MUPMAI->setAllowDotIsPC(false);
  setupCallToAsmParser(AsmStr);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  const MCExpr *Expr;
  bool ParsePrimaryExpr = Parser->parseExpression(Expr);
  EXPECT_EQ(ParsePrimaryExpr, true);
  EXPECT_EQ(Parser->hasPendingError(), true);
}

TEST_F(SystemZAsmLexerTest, CheckRejectStarAsCurrentPC) {
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

TEST_F(SystemZAsmLexerTest, CheckRejectCharLiterals) {
  StringRef AsmStr = "abc 'd'";

  // Setup.
  setupCallToAsmParser(AsmStr);
  Parser->getLexer().setLexHLASMStrings(true);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::Error, AsmToken::Error,
       AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckRejectStringLiterals) {
  StringRef AsmStr = "abc \"ef\"";

  // Setup.
  setupCallToAsmParser(AsmStr);
  Parser->getLexer().setLexHLASMStrings(true);

  // Lex initially to get the string.
  Parser->getLexer().Lex();

  SmallVector<AsmToken::TokenKind> ExpectedTokens(
      {AsmToken::Identifier, AsmToken::Error, AsmToken::Identifier,
       AsmToken::Error, AsmToken::EndOfStatement, AsmToken::Eof});
  lexAndCheckTokens(AsmStr, ExpectedTokens);
}

TEST_F(SystemZAsmLexerTest, CheckPrintAcceptableSymbol) {
  std::string AsmStr = "ab13_$.@";
  EXPECT_EQ(true, MUPMAI->isValidUnquotedName(AsmStr));
  AsmStr += "#";
  EXPECT_EQ(false, MUPMAI->isValidUnquotedName(AsmStr));
}

TEST_F(SystemZAsmLexerTest, CheckPrintAcceptableSymbol2) {
  MUPMAI->setAssemblerDialect(1);
  std::string AsmStr = "ab13_$.@";
  EXPECT_EQ(true, MUPMAI->isValidUnquotedName(AsmStr));
  AsmStr += "#";
  EXPECT_EQ(true, MUPMAI->isValidUnquotedName(AsmStr));
}

TEST_F(SystemZAsmLexerTest, CheckLabelCaseUpperCase2) {
  StringRef AsmStr = "label\nlabel";

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

  // Lex the end of statement token.
  Parser->getLexer().Lex();

  MUPMAI->setEmitLabelsInUpperCase(true);

  ParsePrimaryExpr = Parser->parseExpression(Expr);
  EXPECT_EQ(ParsePrimaryExpr, false);

  SymbolExpr = dyn_cast<MCSymbolRefExpr>(Expr);
  EXPECT_NE(SymbolExpr, nullptr);
  EXPECT_NE(&SymbolExpr->getSymbol(), nullptr);
  EXPECT_EQ((&SymbolExpr->getSymbol())->getName(), StringRef("LABEL"));
}
} // end anonymous namespace
