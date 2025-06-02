//===- AsmLexer.h - Lexer for Assembly Files --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class declares the lexer for assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCPARSER_ASMLEXER_H
#define LLVM_MC_MCPARSER_ASMLEXER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmMacro.h"
#include <cassert>
#include <cstddef>
#include <string>
#include <utility>

namespace llvm {

class MCAsmInfo;

/// A callback class which is notified of each comment in an assembly file as
/// it is lexed.
class AsmCommentConsumer {
public:
  virtual ~AsmCommentConsumer() = default;

  /// Callback function for when a comment is lexed. Loc is the start of the
  /// comment text (excluding the comment-start marker). CommentText is the text
  /// of the comment, excluding the comment start and end markers, and the
  /// newline for single-line comments.
  virtual void HandleComment(SMLoc Loc, StringRef CommentText) = 0;
};

class AsmLexer {
  /// The current token, stored in the base class for faster access.
  SmallVector<AsmToken, 1> CurTok;

  const char *CurPtr = nullptr;
  StringRef CurBuf;

  /// The location and description of the current error
  SMLoc ErrLoc;
  std::string Err;

  const MCAsmInfo &MAI;

  bool IsAtStartOfLine = true;
  bool JustConsumedEOL = true;
  bool IsPeeking = false;
  bool EndStatementAtEOF = true;

  const char *TokStart = nullptr;
  bool SkipSpace = true;
  bool AllowAtInIdentifier = false;
  bool AllowHashInIdentifier = false;
  bool IsAtStartOfStatement = true;
  bool LexMasmHexFloats = false;
  bool LexMasmIntegers = false;
  bool LexMasmStrings = false;
  bool LexMotorolaIntegers = false;
  bool UseMasmDefaultRadix = false;
  unsigned DefaultRadix = 10;
  bool LexHLASMIntegers = false;
  bool LexHLASMStrings = false;
  AsmCommentConsumer *CommentConsumer = nullptr;

  AsmToken LexToken();

  void SetError(SMLoc errLoc, const std::string &err) {
    ErrLoc = errLoc;
    Err = err;
  }

public:
  AsmLexer(const MCAsmInfo &MAI);
  AsmLexer(const AsmLexer &) = delete;
  AsmLexer &operator=(const AsmLexer &) = delete;

  /// Consume the next token from the input stream and return it.
  ///
  /// The lexer will continuously return the end-of-file token once the end of
  /// the main input file has been reached.
  const AsmToken &Lex() {
    assert(!CurTok.empty());
    // Mark if we parsing out a EndOfStatement.
    JustConsumedEOL = CurTok.front().getKind() == AsmToken::EndOfStatement;
    CurTok.erase(CurTok.begin());
    // LexToken may generate multiple tokens via UnLex but will always return
    // the first one. Place returned value at head of CurTok vector.
    if (CurTok.empty()) {
      AsmToken T = LexToken();
      CurTok.insert(CurTok.begin(), T);
    }
    return CurTok.front();
  }

  void UnLex(AsmToken const &Token) {
    CurTok.insert(CurTok.begin(), Token);
  }

  bool justConsumedEOL() { return JustConsumedEOL; }

  StringRef LexUntilEndOfStatement();

  /// Get the current source location.
  SMLoc getLoc() const { return SMLoc::getFromPointer(TokStart); }

  /// Get the current (last) lexed token.
  const AsmToken &getTok() const { return CurTok[0]; }

  /// Look ahead at the next token to be lexed.
  const AsmToken peekTok(bool ShouldSkipSpace = true) {
    AsmToken Tok;

    MutableArrayRef<AsmToken> Buf(Tok);
    size_t ReadCount = peekTokens(Buf, ShouldSkipSpace);

    assert(ReadCount == 1);
    (void)ReadCount;

    return Tok;
  }

  /// Look ahead an arbitrary number of tokens.
  size_t peekTokens(MutableArrayRef<AsmToken> Buf, bool ShouldSkipSpace = true);

  /// Get the current error location
  SMLoc getErrLoc() { return ErrLoc; }

  /// Get the current error string
  const std::string &getErr() { return Err; }

  /// Get the kind of current token.
  AsmToken::TokenKind getKind() const { return getTok().getKind(); }

  /// Check if the current token has kind \p K.
  bool is(AsmToken::TokenKind K) const { return getTok().is(K); }

  /// Check if the current token has kind \p K.
  bool isNot(AsmToken::TokenKind K) const { return getTok().isNot(K); }

  /// Set whether spaces should be ignored by the lexer
  void setSkipSpace(bool val) { SkipSpace = val; }

  bool getAllowAtInIdentifier() { return AllowAtInIdentifier; }
  void setAllowAtInIdentifier(bool v) { AllowAtInIdentifier = v; }

  void setAllowHashInIdentifier(bool V) { AllowHashInIdentifier = V; }

  void setCommentConsumer(AsmCommentConsumer *CommentConsumer) {
    this->CommentConsumer = CommentConsumer;
  }

  /// Set whether to lex masm-style binary (e.g., 0b1101) and radix-specified
  /// literals (e.g., 0ABCh [hex], 576t [decimal], 77o [octal], 1101y [binary]).
  void setLexMasmIntegers(bool V) { LexMasmIntegers = V; }

  /// Set whether to use masm-style default-radix integer literals. If disabled,
  /// assume decimal unless prefixed (e.g., 0x2c [hex], 077 [octal]).
  void useMasmDefaultRadix(bool V) { UseMasmDefaultRadix = V; }

  unsigned getMasmDefaultRadix() const { return DefaultRadix; }
  void setMasmDefaultRadix(unsigned Radix) { DefaultRadix = Radix; }

  /// Set whether to lex masm-style hex float literals, such as 3f800000r.
  void setLexMasmHexFloats(bool V) { LexMasmHexFloats = V; }

  /// Set whether to lex masm-style string literals, such as 'Can''t find file'
  /// and "This ""value"" not found".
  void setLexMasmStrings(bool V) { LexMasmStrings = V; }

  /// Set whether to lex Motorola-style integer literals, such as $deadbeef or
  /// %01010110.
  void setLexMotorolaIntegers(bool V) { LexMotorolaIntegers = V; }

  /// Set whether to lex HLASM-flavour integers. For now this is only [0-9]*
  void setLexHLASMIntegers(bool V) { LexHLASMIntegers = V; }

  /// Set whether to "lex" HLASM-flavour character and string literals. For now,
  /// setting this option to true, will disable lexing for character and string
  /// literals.
  void setLexHLASMStrings(bool V) { LexHLASMStrings = V; }

  void setBuffer(StringRef Buf, const char *ptr = nullptr,
                 bool EndStatementAtEOF = true);

  const MCAsmInfo &getMAI() const { return MAI; }

private:
  bool isAtStartOfComment(const char *Ptr);
  bool isAtStatementSeparator(const char *Ptr);
  [[nodiscard]] int getNextChar();
  int peekNextChar();
  AsmToken ReturnError(const char *Loc, const std::string &Msg);

  AsmToken LexIdentifier();
  AsmToken LexSlash();
  AsmToken LexLineComment();
  AsmToken LexDigit();
  AsmToken LexSingleQuote();
  AsmToken LexQuote();
  AsmToken LexFloatLiteral();
  AsmToken LexHexFloatLiteral(bool NoIntDigits);

  StringRef LexUntilEndOfLine();
};

} // end namespace llvm

#endif // LLVM_MC_MCPARSER_ASMLEXER_H
