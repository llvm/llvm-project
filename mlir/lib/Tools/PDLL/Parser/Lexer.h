//===- Lexer.h - MLIR PDLL Frontend Lexer -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TOOLS_PDLL_PARSER_LEXER_H_
#define LIB_TOOLS_PDLL_PARSER_LEXER_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
struct LogicalResult;

namespace pdll {
namespace ast {
class DiagnosticEngine;
} // namespace ast

//===----------------------------------------------------------------------===//
// Token
//===----------------------------------------------------------------------===//

class Token {
public:
  enum Kind {
    // Markers.
    eof,
    error,

    // Keywords.
    KW_BEGIN,
    // Dependent keywords, i.e. those that are treated as keywords depending on
    // the current parser context.
    KW_DEPENDENT_BEGIN,
    kw_attr,
    kw_op,
    kw_type,
    KW_DEPENDENT_END,

    // General keywords.
    kw_Attr,
    kw_erase,
    kw_let,
    kw_Constraint,
    kw_Op,
    kw_OpName,
    kw_Pattern,
    kw_replace,
    kw_rewrite,
    kw_Type,
    kw_TypeRange,
    kw_Value,
    kw_ValueRange,
    kw_with,
    KW_END,

    // Punctuation.
    arrow,
    colon,
    comma,
    dot,
    equal,
    equal_arrow,
    semicolon,
    // Paired punctuation.
    less,
    greater,
    l_brace,
    r_brace,
    l_paren,
    r_paren,
    l_square,
    r_square,
    underscore,

    // Tokens.
    directive,
    identifier,
    integer,
    string_block,
    string
  };
  Token(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  /// Given a token containing a string literal, return its value, including
  /// removing the quote characters and unescaping the contents of the string.
  std::string getStringValue() const;

  /// Returns true if the current token is a string literal.
  bool isString() const { return isAny(Token::string, Token::string_block); }

  /// Returns true if the current token is a keyword.
  bool isKeyword() const {
    return kind > Token::KW_BEGIN && kind < Token::KW_END;
  }

  /// Returns true if the current token is a keyword in a dependent context, and
  /// in any other situation (e.g. variable names) may be treated as an
  /// identifier.
  bool isDependentKeyword() const {
    return kind > Token::KW_DEPENDENT_BEGIN && kind < Token::KW_DEPENDENT_END;
  }

  /// Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  /// Return the kind of this token.
  Kind getKind() const { return kind; }

  /// Return true if this token is one of the specified kinds.
  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }
  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    return is(k1) || isAny(k2, k3, others...);
  }

  /// Return if the token does not have the given kind.
  bool isNot(Kind k) const { return k != kind; }
  template <typename... T> bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return if the token has the given kind.
  bool is(Kind k) const { return kind == k; }

  /// Return a location for the start of this token.
  SMLoc getStartLoc() const {
    return SMLoc::getFromPointer(spelling.data());
  }
  /// Return a location at the end of this token.
  SMLoc getEndLoc() const {
    return SMLoc::getFromPointer(spelling.data() + spelling.size());
  }
  /// Return a location for the range of this token.
  SMRange getLoc() const {
    return SMRange(getStartLoc(), getEndLoc());
  }

private:
  /// Discriminator that indicates the kind of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

class Lexer {
public:
  Lexer(llvm::SourceMgr &mgr, ast::DiagnosticEngine &diagEngine);
  ~Lexer();

  /// Return a reference to the source manager used by the lexer.
  llvm::SourceMgr &getSourceMgr() { return srcMgr; }

  /// Return a reference to the diagnostic engine used by the lexer.
  ast::DiagnosticEngine &getDiagEngine() { return diagEngine; }

  /// Push an include of the given file. This will cause the lexer to start
  /// processing the provided file. Returns failure if the file could not be
  /// opened, success otherwise.
  LogicalResult pushInclude(StringRef filename);

  /// Lex the next token and return it.
  Token lexToken();

  /// Change the position of the lexer cursor. The next token we lex will start
  /// at the designated point in the input.
  void resetPointer(const char *newPointer) { curPtr = newPointer; }

  /// Emit an error to the lexer with the given location and message.
  Token emitError(SMRange loc, const Twine &msg);
  Token emitError(const char *loc, const Twine &msg);
  Token emitErrorAndNote(SMRange loc, const Twine &msg,
                         SMRange noteLoc, const Twine &note);

private:
  Token formToken(Token::Kind kind, const char *tokStart) {
    return Token(kind, StringRef(tokStart, curPtr - tokStart));
  }

  /// Return the next character in the stream.
  int getNextChar();

  /// Lex methods.
  void lexComment();
  Token lexDirective(const char *tokStart);
  Token lexIdentifier(const char *tokStart);
  Token lexNumber(const char *tokStart);
  Token lexString(const char *tokStart, bool isStringBlock);

  llvm::SourceMgr &srcMgr;
  int curBufferID;
  StringRef curBuffer;
  const char *curPtr;

  /// The engine used to emit diagnostics during lexing/parsing.
  ast::DiagnosticEngine &diagEngine;

  /// A flag indicating if we added a default diagnostic handler to the provided
  /// diagEngine.
  bool addedHandlerToDiagEngine;
};
} // namespace pdll
} // namespace mlir

#endif // LIB_TOOLS_PDLL_PARSER_LEXER_H_
