//===- Lexer.h - Presburger Lexer Interface ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Presburger Lexer class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PARSER_LEXER_H
#define MLIR_ANALYSIS_PRESBURGER_PARSER_LEXER_H

#include "Token.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir::presburger {
/// This class breaks up the current file into a token stream.
class Lexer {
public:
  explicit Lexer(const llvm::SourceMgr &sourceMgr);

  Token lexToken();

  /// Change the position of the lexer cursor.  The next token we lex will start
  /// at the designated point in the input.
  void resetPointer(const char *newPointer) { curPtr = newPointer; }

  /// Returns the start of the buffer.
  const char *getBufferBegin() { return curBuffer.data(); }

private:
  Token formToken(Token::Kind kind, const char *tokStart) {
    return Token(kind, StringRef(tokStart, curPtr - tokStart));
  }

  Token emitError(const char *loc, const llvm::Twine &message);

  // Lexer implementation methods.
  Token lexBareIdentifierOrKeyword(const char *tokStart);
  Token lexNumber(const char *tokStart);

  const llvm::SourceMgr &sourceMgr;

  StringRef curBuffer;
  const char *curPtr;

  Lexer(const Lexer &) = delete;
  void operator=(const Lexer &) = delete;
};
} // namespace mlir::presburger

#endif // MLIR_ANALYSIS_PRESBURGER_PARSER_LEXER_H
