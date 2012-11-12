//===-- X86AsmLexer.cpp - Tokenize X86 assembly to AsmTokens --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCTargetAsmLexer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

namespace {

class X86AsmLexer : public MCTargetAsmLexer {
  const MCAsmInfo &AsmInfo;

  bool tentativeIsValid;
  AsmToken tentativeToken;

  const AsmToken &lexTentative() {
    tentativeToken = getLexer()->Lex();
    tentativeIsValid = true;
    return tentativeToken;
  }

  const AsmToken &lexDefinite() {
    if (tentativeIsValid) {
      tentativeIsValid = false;
      return tentativeToken;
    }
    return getLexer()->Lex();
  }

  AsmToken LexTokenATT();
  AsmToken LexTokenIntel();
protected:
  AsmToken LexToken() {
    if (!Lexer) {
      SetError(SMLoc(), "No MCAsmLexer installed");
      return AsmToken(AsmToken::Error, "", 0);
    }

    switch (AsmInfo.getAssemblerDialect()) {
    default:
      SetError(SMLoc(), "Unhandled dialect");
      return AsmToken(AsmToken::Error, "", 0);
    case 0:
      return LexTokenATT();
    case 1:
      return LexTokenIntel();
    }
  }
public:
  X86AsmLexer(const Target &T, const MCRegisterInfo &MRI, const MCAsmInfo &MAI)
    : MCTargetAsmLexer(T), AsmInfo(MAI), tentativeIsValid(false) {
  }
};

} // end anonymous namespace

#define GET_REGISTER_MATCHER
#include "X86GenAsmMatcher.inc"

AsmToken X86AsmLexer::LexTokenATT() {
  AsmToken lexedToken = lexDefinite();

  switch (lexedToken.getKind()) {
  default:
    return lexedToken;
  case AsmToken::Error:
    SetError(Lexer->getErrLoc(), Lexer->getErr());
    return lexedToken;

  case AsmToken::Percent: {
    const AsmToken &nextToken = lexTentative();
    if (nextToken.getKind() != AsmToken::Identifier)
      return lexedToken;

    if (unsigned regID = MatchRegisterName(nextToken.getString())) {
      lexDefinite();

      // FIXME: This is completely wrong when there is a space or other
      // punctuation between the % and the register name.
      StringRef regStr(lexedToken.getString().data(),
                       lexedToken.getString().size() +
                       nextToken.getString().size());

      return AsmToken(AsmToken::Register, regStr,
                      static_cast<int64_t>(regID));
    }

    // Match register name failed.  If this is "db[0-7]", match it as an alias
    // for dr[0-7].
    if (nextToken.getString().size() == 3 &&
        nextToken.getString().startswith("db")) {
      int RegNo = -1;
      switch (nextToken.getString()[2]) {
      case '0': RegNo = X86::DR0; break;
      case '1': RegNo = X86::DR1; break;
      case '2': RegNo = X86::DR2; break;
      case '3': RegNo = X86::DR3; break;
      case '4': RegNo = X86::DR4; break;
      case '5': RegNo = X86::DR5; break;
      case '6': RegNo = X86::DR6; break;
      case '7': RegNo = X86::DR7; break;
      }

      if (RegNo != -1) {
        lexDefinite();

        // FIXME: This is completely wrong when there is a space or other
        // punctuation between the % and the register name.
        StringRef regStr(lexedToken.getString().data(),
                         lexedToken.getString().size() +
                         nextToken.getString().size());
        return AsmToken(AsmToken::Register, regStr,
                        static_cast<int64_t>(RegNo));
      }
    }


    return lexedToken;
  }
  }
}

AsmToken X86AsmLexer::LexTokenIntel() {
  const AsmToken &lexedToken = lexDefinite();

  switch(lexedToken.getKind()) {
  default:
    return lexedToken;
  case AsmToken::Error:
    SetError(Lexer->getErrLoc(), Lexer->getErr());
    return lexedToken;
  case AsmToken::Identifier: {
    unsigned regID = MatchRegisterName(lexedToken.getString().lower());

    if (regID)
      return AsmToken(AsmToken::Register,
                      lexedToken.getString(),
                      static_cast<int64_t>(regID));
    return lexedToken;
  }
  }
}

extern "C" void LLVMInitializeX86AsmLexer() {
  RegisterMCAsmLexer<X86AsmLexer> X(TheX86_32Target);
  RegisterMCAsmLexer<X86AsmLexer> Y(TheX86_64Target);
}
