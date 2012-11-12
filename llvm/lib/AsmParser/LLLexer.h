//===- LLLexer.h - Lexer for LLVM Assembly Files ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents the Lexer for .ll files.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_ASMPARSER_LLLEXER_H
#define LIB_ASMPARSER_LLLEXER_H

#include "LLToken.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/SourceMgr.h"
#include <string>

namespace llvm {
  class MemoryBuffer;
  class Type;
  class SMDiagnostic;
  class LLVMContext;

  class LLLexer {
    const char *CurPtr;
    MemoryBuffer *CurBuf;
    SMDiagnostic &ErrorInfo;
    SourceMgr &SM;
    LLVMContext &Context;

    // Information about the current token.
    const char *TokStart;
    lltok::Kind CurKind;
    std::string StrVal;
    unsigned UIntVal;
    Type *TyVal;
    APFloat APFloatVal;
    APSInt  APSIntVal;

  public:
    explicit LLLexer(MemoryBuffer *StartBuf, SourceMgr &SM, SMDiagnostic &,
                     LLVMContext &C);
    ~LLLexer() {}

    lltok::Kind Lex() {
      return CurKind = LexToken();
    }

    typedef SMLoc LocTy;
    LocTy getLoc() const { return SMLoc::getFromPointer(TokStart); }
    lltok::Kind getKind() const { return CurKind; }
    const std::string &getStrVal() const { return StrVal; }
    Type *getTyVal() const { return TyVal; }
    unsigned getUIntVal() const { return UIntVal; }
    const APSInt &getAPSIntVal() const { return APSIntVal; }
    const APFloat &getAPFloatVal() const { return APFloatVal; }


    bool Error(LocTy L, const Twine &Msg) const;
    bool Error(const Twine &Msg) const { return Error(getLoc(), Msg); }
    std::string getFilename() const;

  private:
    lltok::Kind LexToken();

    int getNextChar();
    void SkipLineComment();
    lltok::Kind ReadString(lltok::Kind kind);
    bool ReadVarName();

    lltok::Kind LexIdentifier();
    lltok::Kind LexDigitOrNegative();
    lltok::Kind LexPositive();
    lltok::Kind LexAt();
    lltok::Kind LexExclaim();
    lltok::Kind LexPercent();
    lltok::Kind LexQuote();
    lltok::Kind Lex0x();

    uint64_t atoull(const char *Buffer, const char *End);
    uint64_t HexIntToVal(const char *Buffer, const char *End);
    void HexToIntPair(const char *Buffer, const char *End, uint64_t Pair[2]);
    void FP80HexToIntPair(const char *Buff, const char *End, uint64_t Pair[2]);
  };
} // end namespace llvm

#endif
