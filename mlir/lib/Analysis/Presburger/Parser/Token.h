//===- Token.h - Presburger Token Interface ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PARSER_TOKEN_H
#define MLIR_ANALYSIS_PRESBURGER_PARSER_TOKEN_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include <optional>

namespace mlir::presburger {
using llvm::SMLoc;
using llvm::SMRange;
using llvm::StringRef;

class Token {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "TokenKinds.def"
  };

  Token(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  // Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  // Token classification.
  Kind getKind() const { return kind; }
  bool is(Kind k) const { return kind == k; }

  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }

  /// Return true if this token is one of the specified kinds.
  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  bool isNot(Kind k) const { return kind != k; }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return true if this is one of the keyword token kinds (e.g. kw_if).
  bool isKeyword() const;

  // Helpers to decode specific sorts of tokens.

  /// For an integer token, return its value as an uint64_t.  If it doesn't fit,
  /// return std::nullopt.
  static std::optional<uint64_t> getUInt64IntegerValue(StringRef spelling);
  std::optional<uint64_t> getUInt64IntegerValue() const {
    return getUInt64IntegerValue(getSpelling());
  }

  // Location processing.
  SMLoc getLoc() const;
  SMLoc getEndLoc() const;
  SMRange getLocRange() const;

  /// Given a punctuation or keyword token kind, return the spelling of the
  /// token as a string.  Warning: This will abort on markers, identifiers and
  /// literal tokens since they have no fixed spelling.
  static StringRef getTokenSpelling(Kind kind);

private:
  /// Discriminator that indicates the sort of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};
} // namespace mlir::presburger

#endif // MLIR_ANALYSIS_PRESBURGER_PARSER_TOKEN_H
