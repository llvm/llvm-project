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
#include <vector>
#include <string>
#include <iosfwd>
#include <cassert>

namespace llvm {
class MemoryBuffer;
  
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
    colon, semi,        // ; :
    comma, period,      // , .
    equal, question,    // = ?
    
    // Keywords.
    Bit, Bits, Class, Code, Dag, Def, Defm, Field, In, Int, Let, List,
    MultiClass, String,
    
    // !keywords.
    XConcat, XSRA, XSRL, XSHL, XStrConcat,
    
    // Integer value.
    IntVal,
    
    // String valued tokens.
    Id, StrVal, VarName, CodeFragment
  };
}

/// TGLexer - TableGen Lexer class.
class TGLexer {
  const char *CurPtr;
  unsigned CurLineNo;
  MemoryBuffer *CurBuf;

  // Information about the current token.
  const char *TokStart;
  tgtok::TokKind CurCode;
  std::string CurStrVal;  // This is valid for ID, STRVAL, VARNAME, CODEFRAGMENT
  int64_t CurIntVal;      // This is valid for INTVAL.
  
  /// IncludeRec / IncludeStack - This captures the current set of include
  /// directives we are nested within.
  struct IncludeRec {
    MemoryBuffer *Buffer;
    const char *CurPtr;
    unsigned LineNo;
    IncludeRec(MemoryBuffer *buffer, const char *curPtr, unsigned lineNo)
      : Buffer(buffer), CurPtr(curPtr), LineNo(lineNo) {}
  };
  std::vector<IncludeRec> IncludeStack;
  
  // IncludeDirectories - This is the list of directories we should search for
  // include files in.
  std::vector<std::string> IncludeDirectories;
public:
  TGLexer(MemoryBuffer *StartBuf);
  ~TGLexer();
  
  void setIncludeDirs(const std::vector<std::string> &Dirs) {
    IncludeDirectories = Dirs;
  }
  
  tgtok::TokKind Lex() {
    return CurCode = LexToken();
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

  typedef const char* LocTy;
  LocTy getLoc() const { return TokStart; }

  void PrintError(LocTy Loc, const std::string &Msg) const;
  
  void PrintIncludeStack(std::ostream &OS) const;
  
private:
  /// LexToken - Read the next token and return its code.
  tgtok::TokKind LexToken();
  
  tgtok::TokKind ReturnError(const char *Loc, const std::string &Msg);
  
  int getNextChar();
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
