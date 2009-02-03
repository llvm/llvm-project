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
#include <string>

namespace llvm {
  class MemoryBuffer;
  class Type;
  class ParseError;

  class LLLexer {
    const char *CurPtr;
    MemoryBuffer *CurBuf;
    ParseError &ErrorInfo;

    // Information about the current token.
    const char *TokStart;
    lltok::Kind CurKind;
    std::string StrVal;
    unsigned UIntVal;
    const Type *TyVal;
    APFloat APFloatVal;
    APSInt  APSIntVal;

    std::string TheError;
  public:
    explicit LLLexer(MemoryBuffer *StartBuf, ParseError &);
    ~LLLexer() {}

    lltok::Kind Lex() {
      return CurKind = LexToken();
    }

    typedef const char* LocTy;
    LocTy getLoc() const { return TokStart; }
    lltok::Kind getKind() const { return CurKind; }
    const std::string getStrVal() const { return StrVal; }
    const Type *getTyVal() const { return TyVal; }
    unsigned getUIntVal() const { return UIntVal; }
    const APSInt &getAPSIntVal() const { return APSIntVal; }
    const APFloat &getAPFloatVal() const { return APFloatVal; }


    bool Error(LocTy L, const std::string &Msg) const;
    bool Error(const std::string &Msg) const { return Error(CurPtr, Msg); }
    std::string getFilename() const;

  private:
    lltok::Kind LexToken();

    int getNextChar();
    void SkipLineComment();
    lltok::Kind LexIdentifier();
    lltok::Kind LexDigitOrNegative();
    lltok::Kind LexPositive();
    lltok::Kind LexAt();
    lltok::Kind LexPercent();
    lltok::Kind LexQuote();
    lltok::Kind Lex0x();

    uint64_t atoull(const char *Buffer, const char *End);
    uint64_t HexIntToVal(const char *Buffer, const char *End);
    void HexToIntPair(const char *Buffer, const char *End, uint64_t Pair[2]);
  };
} // end namespace llvm

#endif
