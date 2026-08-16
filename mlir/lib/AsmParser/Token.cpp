//===- Token.cpp - MLIR Token Implementation ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Token class for the MLIR textual form.
//
//===----------------------------------------------------------------------===//

#include "Token.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

using namespace mlir;

SMLoc Token::getLoc() const { return SMLoc::getFromPointer(spelling.data()); }

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange Token::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

/// For an integer token, return its value as an unsigned.  If it doesn't fit,
/// return std::nullopt.
std::optional<unsigned> Token::getUnsignedIntegerValue() const {
  bool isHex = spelling.size() > 1 && spelling[1] == 'x';

  unsigned result = 0;
  if (spelling.getAsInteger(isHex ? 0 : 10, result))
    return std::nullopt;
  return result;
}

/// For an integer token, return its value as a uint64_t.  If it doesn't fit,
/// return std::nullopt.
std::optional<uint64_t> Token::getUInt64IntegerValue(StringRef spelling) {
  bool isHex = spelling.size() > 1 && spelling[1] == 'x';

  uint64_t result = 0;
  if (spelling.getAsInteger(isHex ? 0 : 10, result))
    return std::nullopt;
  return result;
}

/// For a floatliteral token, build its value in `semantics`, combining any sign
/// folded into the spelling with `isNegative`. On failure, emits a diagnostic
/// through `emitError` and returns std::nullopt.
std::optional<APFloat> Token::getFloatingPointValue(
    bool isNegative, const llvm::fltSemantics &semantics,
    function_ref<InFlightDiagnostic()> emitError) const {
  // The sign is either a preceding '-' token (isNegative) or folded into an
  // inf/NaN spelling; both at once (e.g. `-+inf`) is invalid. Drop a leading
  // '+' since convertFromString rejects it on NaN forms.
  StringRef str = spelling;
  if (isNegative && (str.starts_with("+") || str.starts_with("-"))) {
    emitError() << "floating point literal has more than one sign";
    return std::nullopt;
  }
  bool isNeg = isNegative || str.consume_front("-");
  str.consume_front("+");

  // Reject values the type cannot represent; APFloat would otherwise abort.
  if (isNeg && !APFloat::semanticsHasSignedRepr(semantics)) {
    emitError() << "floating point type does not support negative values";
    return std::nullopt;
  }
  if (str == "inf" && !APFloat::semanticsHasInf(semantics)) {
    emitError() << "floating point type does not support infinity";
    return std::nullopt;
  }
  bool wantsNaN =
      str == "qnan" || str.starts_with("nan") || str.starts_with("snan");
  if (wantsNaN && !APFloat::semanticsHasNaN(semantics)) {
    emitError() << "floating point type does not support NaN";
    return std::nullopt;
  }

  // NanOnly types have a single NaN per representable sign, with no payload and
  // no quiet/signaling distinction, and the negative-zero encoding has only a
  // negative NaN. Reject spellings the type cannot represent instead of
  // silently re-encoding them.
  if (wantsNaN &&
      semantics.nonFiniteBehavior == llvm::fltNonfiniteBehavior::NanOnly) {
    if (str.starts_with("snan")) {
      emitError() << "floating point type does not support signaling NaN";
      return std::nullopt;
    }
    if (str.contains('(')) {
      emitError() << "floating point type does not support NaN payload";
      return std::nullopt;
    }
    // The negative-zero encoding has only a negative NaN; a positive one is not
    // representable and would silently yield the negative bit pattern.
    if (!isNeg && semantics.nanEncoding == llvm::fltNanEncoding::NegativeZero) {
      emitError() << "floating point type only supports negative NaN";
      return std::nullopt;
    }
  }

  // convertFromString does not accept "qnan".
  if (str == "qnan")
    return APFloat::getQNaN(semantics, isNeg);

  // Build in the target semantics to preserve NaN payloads; overflow/underflow
  // are tolerated (inf/zero).
  APFloat result(semantics);
  llvm::Expected<APFloat::opStatus> status =
      result.convertFromString(str, APFloat::rmNearestTiesToEven);
  if (!status) {
    llvm::consumeError(status.takeError());
    emitError() << "invalid floating point literal";
    return std::nullopt;
  }
  if (isNeg)
    result.changeSign();
  return result;
}

/// For an inttype token, return its bitwidth.
std::optional<unsigned> Token::getIntTypeBitwidth() const {
  assert(getKind() == inttype);
  unsigned bitwidthStart = (spelling[0] == 'i' ? 1 : 2);
  unsigned result = 0;
  if (spelling.drop_front(bitwidthStart).getAsInteger(10, result))
    return std::nullopt;
  return result;
}

std::optional<bool> Token::getIntTypeSignedness() const {
  assert(getKind() == inttype);
  if (spelling[0] == 'i')
    return std::nullopt;
  if (spelling[0] == 's')
    return true;
  assert(spelling[0] == 'u');
  return false;
}

/// Given a token containing a string literal, return its value, including
/// removing the quote characters and unescaping the contents of the string. The
/// lexer has already verified that this token is valid.
std::string Token::getStringValue() const {
  assert(getKind() == string || getKind() == code_complete ||
         (getKind() == at_identifier && getSpelling()[1] == '"'));
  // Start by dropping the quotes.
  StringRef bytes = getSpelling().drop_front();
  if (getKind() != Token::code_complete) {
    bytes = bytes.drop_back();
    if (getKind() == at_identifier)
      bytes = bytes.drop_front();
  }

  std::string result;
  result.reserve(bytes.size());
  for (unsigned i = 0, e = bytes.size(); i != e;) {
    auto c = bytes[i++];
    if (c != '\\') {
      result.push_back(c);
      continue;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c1 = bytes[i++];
    switch (c1) {
    case '"':
    case '\\':
      result.push_back(c1);
      continue;
    case 'n':
      result.push_back('\n');
      continue;
    case 't':
      result.push_back('\t');
      continue;
    default:
      break;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c2 = bytes[i++];

    assert(llvm::isHexDigit(c1) && llvm::isHexDigit(c2) && "invalid escape");
    result.push_back((llvm::hexDigitValue(c1) << 4) | llvm::hexDigitValue(c2));
  }

  return result;
}

/// Given a token containing a hex string literal, return its value or
/// std::nullopt if the token does not contain a valid hex string.
std::optional<std::string> Token::getHexStringValue() const {
  assert(getKind() == string);

  // Get the internal string data, without the quotes.
  StringRef bytes = getSpelling().drop_front().drop_back();

  // Try to extract the binary data from the hex string. We expect the hex
  // string to start with `0x` and have an even number of hex nibbles (nibbles
  // should come in pairs).
  std::string hex;
  if (!bytes.consume_front("0x") || (bytes.size() & 1) ||
      !llvm::tryGetFromHex(bytes, hex))
    return std::nullopt;
  return hex;
}

/// Given a token containing a symbol reference, return the unescaped string
/// value.
std::string Token::getSymbolReference() const {
  assert(is(Token::at_identifier) && "expected valid @-identifier");
  StringRef nameStr = getSpelling().drop_front();

  // Check to see if the reference is a string literal, or a bare identifier.
  if (nameStr.front() == '"')
    return getStringValue();
  return std::string(nameStr);
}

/// Given a hash_identifier token like #123, try to parse the number out of
/// the identifier, returning std::nullopt if it is a named identifier like #x
/// or if the integer doesn't fit.
std::optional<unsigned> Token::getHashIdentifierNumber() const {
  assert(getKind() == hash_identifier);
  unsigned result = 0;
  if (spelling.drop_front().getAsInteger(10, result))
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

bool Token::isCodeCompletionFor(Kind kind) const {
  if (!isCodeCompletion() || spelling.empty())
    return false;
  switch (kind) {
  case Kind::string:
    return spelling[0] == '"';
  case Kind::hash_identifier:
    return spelling[0] == '#';
  case Kind::percent_identifier:
    return spelling[0] == '%';
  case Kind::caret_identifier:
    return spelling[0] == '^';
  case Kind::exclamation_identifier:
    return spelling[0] == '!';
  default:
    return false;
  }
}
