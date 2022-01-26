//===- FormatGen.h - Utilities for custom assembly formats ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common classes for building custom assembly format parsers
// and generators.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_FORMATGEN_H_
#define MLIR_TOOLS_MLIRTBLGEN_FORMATGEN_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
namespace tblgen {

//===----------------------------------------------------------------------===//
// FormatToken
//===----------------------------------------------------------------------===//

/// This class represents a specific token in the input format.
class FormatToken {
public:
  /// Basic token kinds.
  enum Kind {
    // Markers.
    eof,
    error,

    // Tokens with no info.
    l_paren,
    r_paren,
    caret,
    colon,
    comma,
    equal,
    less,
    greater,
    question,
    star,

    // Keywords.
    keyword_start,
    kw_attr_dict,
    kw_attr_dict_w_keyword,
    kw_custom,
    kw_functional_type,
    kw_operands,
    kw_params,
    kw_qualified,
    kw_ref,
    kw_regions,
    kw_results,
    kw_struct,
    kw_successors,
    kw_type,
    keyword_end,

    // String valued tokens.
    identifier,
    literal,
    variable,
  };

  FormatToken(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  /// Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  /// Return the kind of this token.
  Kind getKind() const { return kind; }

  /// Return a location for this token.
  SMLoc getLoc() const;

  /// Return if this token is a keyword.
  bool isKeyword() const {
    return getKind() > Kind::keyword_start && getKind() < Kind::keyword_end;
  }

private:
  /// Discriminator that indicates the kind of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

//===----------------------------------------------------------------------===//
// FormatLexer
//===----------------------------------------------------------------------===//

/// This class implements a simple lexer for operation assembly format strings.
class FormatLexer {
public:
  FormatLexer(llvm::SourceMgr &mgr, SMLoc loc);

  /// Lex the next token and return it.
  FormatToken lexToken();

  /// Emit an error to the lexer with the given location and message.
  FormatToken emitError(SMLoc loc, const Twine &msg);
  FormatToken emitError(const char *loc, const Twine &msg);

  FormatToken emitErrorAndNote(SMLoc loc, const Twine &msg,
                               const Twine &note);

private:
  /// Return the next character in the stream.
  int getNextChar();

  /// Lex an identifier, literal, or variable.
  FormatToken lexIdentifier(const char *tokStart);
  FormatToken lexLiteral(const char *tokStart);
  FormatToken lexVariable(const char *tokStart);

  /// Create a token with the current pointer and a start pointer.
  FormatToken formToken(FormatToken::Kind kind, const char *tokStart) {
    return FormatToken(kind, StringRef(tokStart, curPtr - tokStart));
  }

  /// The source manager containing the format string.
  llvm::SourceMgr &mgr;
  /// Location of the format string.
  SMLoc loc;
  /// Buffer containing the format string.
  StringRef curBuffer;
  /// Current pointer in the buffer.
  const char *curPtr;
};

/// Whether a space needs to be emitted before a literal. E.g., two keywords
/// back-to-back require a space separator, but a keyword followed by '<' does
/// not require a space.
bool shouldEmitSpaceBefore(StringRef value, bool lastWasPunctuation);

/// Returns true if the given string can be formatted as a keyword.
bool canFormatStringAsKeyword(StringRef value,
                              function_ref<void(Twine)> emitError = nullptr);

/// Returns true if the given string is valid format literal element.
/// If `emitError` is provided, it is invoked with the reason for the failure.
bool isValidLiteral(StringRef value,
                    function_ref<void(Twine)> emitError = nullptr);

/// Whether a failure in parsing the assembly format should be a fatal error.
extern llvm::cl::opt<bool> formatErrorIsFatal;

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_FORMATGEN_H_
