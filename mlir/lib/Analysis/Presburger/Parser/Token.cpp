//===- Token.cpp - Presburger Token Implementation --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Token class for the Presburger textual form.
//
//===----------------------------------------------------------------------===//

#include "Token.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include <optional>

using namespace mlir::presburger;

SMLoc Token::getLoc() const { return SMLoc::getFromPointer(spelling.data()); }

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange Token::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

/// For an integer token, return its value as a uint64_t.  If it doesn't fit,
/// return std::nullopt.
std::optional<uint64_t> Token::getUInt64IntegerValue(StringRef spelling) {
  uint64_t result = 0;
  if (spelling.getAsInteger(10, result))
    return std::nullopt;
  return result;
}

/// Given a punctuation or keyword token kind, return the spelling of the
/// token as a string.  Warning: This will abort on markers, identifiers and
/// literal tokens since they have no fixed spelling.
StringRef Token::getTokenSpelling(Kind kind) {
  switch (kind) {
  default:
    llvm_unreachable("This token kind has no fixed spelling");
#define TOK_PUNCTUATION(NAME, SPELLING)                                        \
  case NAME:                                                                   \
    return SPELLING;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return #SPELLING;
#include "TokenKinds.def"
  }
}

/// Return true if this is one of the keyword token kinds (e.g. kw_if).
bool Token::isKeyword() const {
  switch (kind) {
  default:
    return false;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return true;
#include "TokenKinds.def"
  }
}
