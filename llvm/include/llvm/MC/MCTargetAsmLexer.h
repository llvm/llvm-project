//===-- llvm/MC/MCTargetAsmLexer.h - Target Assembly Lexer ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCTARGETASMLEXER_H
#define LLVM_MC_MCTARGETASMLEXER_H

#include "llvm/MC/MCParser/MCAsmLexer.h"

namespace llvm {
class Target;

/// MCTargetAsmLexer - Generic interface to target specific assembly lexers.
class MCTargetAsmLexer {
  /// The current token
  AsmToken CurTok;

  /// The location and description of the current error
  SMLoc ErrLoc;
  std::string Err;

  MCTargetAsmLexer(const MCTargetAsmLexer &) LLVM_DELETED_FUNCTION;
  void operator=(const MCTargetAsmLexer &) LLVM_DELETED_FUNCTION;
protected: // Can only create subclasses.
  MCTargetAsmLexer(const Target &);

  virtual AsmToken LexToken() = 0;

  void SetError(const SMLoc &errLoc, const std::string &err) {
    ErrLoc = errLoc;
    Err = err;
  }

  /// TheTarget - The Target that this machine was created for.
  const Target &TheTarget;
  MCAsmLexer *Lexer;

public:
  virtual ~MCTargetAsmLexer();

  const Target &getTarget() const { return TheTarget; }

  /// InstallLexer - Set the lexer to get tokens from lower-level lexer \p L.
  void InstallLexer(MCAsmLexer &L) {
    Lexer = &L;
  }

  MCAsmLexer *getLexer() {
    return Lexer;
  }

  /// Lex - Consume the next token from the input stream and return it.
  const AsmToken &Lex() {
    return CurTok = LexToken();
  }

  /// getTok - Get the current (last) lexed token.
  const AsmToken &getTok() {
    return CurTok;
  }

  /// getErrLoc - Get the current error location
  const SMLoc &getErrLoc() {
    return ErrLoc;
  }

  /// getErr - Get the current error string
  const std::string &getErr() {
    return Err;
  }

  /// getKind - Get the kind of current token.
  AsmToken::TokenKind getKind() const { return CurTok.getKind(); }

  /// is - Check if the current token has kind \p K.
  bool is(AsmToken::TokenKind K) const { return CurTok.is(K); }

  /// isNot - Check if the current token has kind \p K.
  bool isNot(AsmToken::TokenKind K) const { return CurTok.isNot(K); }
};

} // End llvm namespace

#endif
