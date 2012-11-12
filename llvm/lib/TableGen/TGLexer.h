//===- TGLexer.h - Lexer for TableGen Files ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents the Lexer for tablegen files.
//
//===----------------------------------------------------------------------===//

#ifndef TGLEXER_H
#define TGLEXER_H

#include "llvm/Support/DataTypes.h"
#include <string>
#include <vector>
#include <cassert>

namespace llvm {
class MemoryBuffer;
class SourceMgr;
class SMLoc;
class Twine;

namespace tgtok {
  enum TokKind {
    // Markers
    Eof, Error,
    
    // Tokens with no info.
    minus, plus,        // - +
    l_square, r_square, // [ ]
    l_brace, r_brace,   // { }
    l_paren, r_paren,   // ( )
    less, greater,      // < >
    colon, semi,        // : ;
    comma, period,      // , .
    equal, question,    // = ?
    paste,              // #

    // Keywords.
    Bit, Bits, Class, Code, Dag, Def, Foreach, Defm, Field, In, Int, Let, List,
    MultiClass, String,
    
    // !keywords.
    XConcat, XSRA, XSRL, XSHL, XStrConcat, XCast, XSubst,
    XForEach, XHead, XTail, XEmpty, XIf, XEq,

    // Integer value.
    IntVal,
    
    // String valued tokens.
    Id, StrVal, VarName, CodeFragment
  };
}

/// TGLexer - TableGen Lexer class.
class TGLexer {
  SourceMgr &SrcMgr;
  
  const char *CurPtr;
  const MemoryBuffer *CurBuf;

  // Information about the current token.
  const char *TokStart;
  tgtok::TokKind CurCode;
  std::string CurStrVal;  // This is valid for ID, STRVAL, VARNAME, CODEFRAGMENT
  int64_t CurIntVal;      // This is valid for INTVAL.

  /// CurBuffer - This is the current buffer index we're lexing from as managed
  /// by the SourceMgr object.
  int CurBuffer;
  /// Dependencies - This is the list of all included files.
  std::vector<std::string> Dependencies;
  
public:
  TGLexer(SourceMgr &SrcMgr);
  ~TGLexer() {}
  
  tgtok::TokKind Lex() {
    return CurCode = LexToken();
  }

  const std::vector<std::string> &getDependencies() const {
    return Dependencies;
  }
  
  tgtok::TokKind getCode() const { return CurCode; }

  const std::string &getCurStrVal() const {
    assert((CurCode == tgtok::Id || CurCode == tgtok::StrVal || 
            CurCode == tgtok::VarName || CurCode == tgtok::CodeFragment) &&
           "This token doesn't have a string value");
    return CurStrVal;
  }
  int64_t getCurIntVal() const {
    assert(CurCode == tgtok::IntVal && "This token isn't an integer");
    return CurIntVal;
  }

  SMLoc getLoc() const;
  
private:
  /// LexToken - Read the next token and return its code.
  tgtok::TokKind LexToken();
  
  tgtok::TokKind ReturnError(const char *Loc, const Twine &Msg);
  
  int getNextChar();
  int peekNextChar(int Index);
  void SkipBCPLComment();
  bool SkipCComment();
  tgtok::TokKind LexIdentifier();
  bool LexInclude();
  tgtok::TokKind LexString();
  tgtok::TokKind LexVarName();
  tgtok::TokKind LexNumber();
  tgtok::TokKind LexBracket();
  tgtok::TokKind LexExclaim();
};
  
} // end namespace llvm

#endif
