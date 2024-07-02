//===- Lexer.cpp - Presburger Lexer Implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lexer for the Presburger textual form.
//
//===----------------------------------------------------------------------===//

#include "Lexer.h"
#include "Token.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir::presburger;

Lexer::Lexer(const llvm::SourceMgr &sourceMgr) : sourceMgr(sourceMgr) {
  auto bufferID = sourceMgr.getMainFileID();
  curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  curPtr = curBuffer.begin();
}

Token Lexer::emitError(const char *loc, const llvm::Twine &message) {
  sourceMgr.PrintMessage(SMLoc::getFromPointer(loc), llvm::SourceMgr::DK_Error,
                         message);
  return formToken(Token::error, loc);
}

Token Lexer::lexToken() {
  while (true) {
    const char *tokStart = curPtr;

    switch (*curPtr++) {
    default:
      // Handle bare identifiers.
      if (isalpha(curPtr[-1]))
        return lexBareIdentifierOrKeyword(tokStart);
      return emitError(tokStart, "unexpected character");

    case ' ':
    case '\t':
    case '\n':
    case '\r':
      // Handle whitespace.
      continue;

    case 0:
      // This may be the EOF marker that llvm::MemoryBuffer guarantees will be
      // there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(Token::eof, tokStart);
      return emitError(tokStart, "unexpected character");

    case ':':
      return formToken(Token::colon, tokStart);
    case ',':
      return formToken(Token::comma, tokStart);
    case '(':
      return formToken(Token::l_paren, tokStart);
    case ')':
      return formToken(Token::r_paren, tokStart);
    case '{':
      return formToken(Token::l_brace, tokStart);
    case '}':
      return formToken(Token::r_brace, tokStart);
    case '[':
      return formToken(Token::l_square, tokStart);
    case ']':
      return formToken(Token::r_square, tokStart);
    case '<':
      return formToken(Token::less, tokStart);
    case '>':
      return formToken(Token::greater, tokStart);
    case '=':
      return formToken(Token::equal, tokStart);
    case '+':
      return formToken(Token::plus, tokStart);
    case '*':
      return formToken(Token::star, tokStart);
    case '-':
      if (*curPtr == '>') {
        ++curPtr;
        return formToken(Token::arrow, tokStart);
      }
      return formToken(Token::minus, tokStart);
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexNumber(tokStart);
    }
  }
}

/// Lex a bare identifier or keyword that starts with a letter.
///
///   bare-id ::= (letter|[_]) (letter|digit|[_$.])*
///
Token Lexer::lexBareIdentifierOrKeyword(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_.$]*
  while (isalpha(*curPtr) || isdigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$' || *curPtr == '.')
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef spelling(tokStart, curPtr - tokStart);

  Token::Kind kind = llvm::StringSwitch<Token::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, Token::kw_##SPELLING)
#include "TokenKinds.def"
                         .Default(Token::bare_identifier);

  return Token(kind, spelling);
}

/// Lex a number literal.
///
///   integer-literal ::= digit+
///
Token Lexer::lexNumber(const char *tokStart) {
  assert(isdigit(curPtr[-1]));

  // Handle the normal decimal case.
  while (isdigit(*curPtr))
    ++curPtr;

  return formToken(Token::integer, tokStart);
}
