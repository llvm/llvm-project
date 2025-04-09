//===- LocationParser.cpp - MLIR Location Parser  -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Parser.h"
#include "Token.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include <optional>

using namespace mlir;
using namespace mlir::detail;

/// Specific location instances.
///
/// location-inst ::= filelinecol-location |
///                   name-location |
///                   callsite-location |
///                   fused-location |
///                   unknown-location
/// filelinecol-location ::= string-literal ':' integer-literal
///                                         ':' integer-literal
/// name-location ::= string-literal
/// callsite-location ::= 'callsite' '(' location-inst 'at' location-inst ')'
/// fused-location ::= fused ('<' attribute-value '>')?
///                    '[' location-inst (location-inst ',')* ']'
/// unknown-location ::= 'unknown'
///
ParseResult Parser::parseCallSiteLocation(LocationAttr &loc) {
  consumeToken(Token::bare_identifier);

  // Parse the '('.
  if (parseToken(Token::l_paren, "expected '(' in callsite location"))
    return failure();

  // Parse the callee location.
  LocationAttr calleeLoc;
  if (parseLocationInstance(calleeLoc))
    return failure();

  // Parse the 'at'.
  if (getToken().isNot(Token::bare_identifier) ||
      getToken().getSpelling() != "at")
    return emitWrongTokenError("expected 'at' in callsite location");
  consumeToken(Token::bare_identifier);

  // Parse the caller location.
  LocationAttr callerLoc;
  if (parseLocationInstance(callerLoc))
    return failure();

  // Parse the ')'.
  if (parseToken(Token::r_paren, "expected ')' in callsite location"))
    return failure();

  // Return the callsite location.
  loc = CallSiteLoc::get(calleeLoc, callerLoc);
  return success();
}

ParseResult Parser::parseFusedLocation(LocationAttr &loc) {
  consumeToken(Token::bare_identifier);

  // Try to parse the optional metadata.
  Attribute metadata;
  if (consumeIf(Token::less)) {
    metadata = parseAttribute();
    if (!metadata)
      return failure();

    // Parse the '>' token.
    if (parseToken(Token::greater,
                   "expected '>' after fused location metadata"))
      return failure();
  }

  SmallVector<Location, 4> locations;
  auto parseElt = [&] {
    LocationAttr newLoc;
    if (parseLocationInstance(newLoc))
      return failure();
    locations.push_back(newLoc);
    return success();
  };

  if (parseCommaSeparatedList(Delimiter::Square, parseElt,
                              " in fused location"))
    return failure();

  // Return the fused location.
  loc = FusedLoc::get(locations, metadata, getContext());
  return success();
}

ParseResult Parser::parseNameOrFileLineColRange(LocationAttr &loc) {
  auto *ctx = getContext();
  auto str = getToken().getStringValue();
  consumeToken(Token::string);

  std::optional<unsigned> startLine, startColumn, endLine, endColumn;

  // If the next token is ':' this is a filelinecol location.
  if (consumeIf(Token::colon)) {
    // Parse the line number.
    if (getToken().isNot(Token::integer))
      return emitWrongTokenError(
          "expected integer line number in FileLineColRange");
    startLine = getToken().getUnsignedIntegerValue();
    if (!startLine)
      return emitWrongTokenError(
          "expected integer line number in FileLineColRange");
    consumeToken(Token::integer);

    // Parse the ':'.
    if (getToken().isNot(Token::colon)) {
      loc = FileLineColRange::get(StringAttr::get(ctx, str), *startLine);
      return success();
    }
    consumeToken(Token::colon);

    // Parse the column number.
    if (getToken().isNot(Token::integer)) {
      return emitWrongTokenError(
          "expected integer column number in FileLineColRange");
    }
    startColumn = getToken().getUnsignedIntegerValue();
    if (!startColumn.has_value())
      return emitError("expected integer column number in FileLineColRange");
    consumeToken(Token::integer);

    if (!isCurrentTokenAKeyword() || getTokenSpelling() != "to") {
      loc = FileLineColLoc::get(ctx, str, *startLine, *startColumn);
      return success();
    }
    consumeToken();

    // Parse the line number.
    if (getToken().is(Token::integer)) {
      endLine = getToken().getUnsignedIntegerValue();
      if (!endLine) {
        return emitWrongTokenError(
            "expected integer line number in FileLineColRange");
      }
      consumeToken(Token::integer);
    }

    // Parse the ':'.
    if (getToken().isNot(Token::colon)) {
      return emitWrongTokenError(
          "expected either integer or `:` post `to` in FileLineColRange");
    }
    consumeToken(Token::colon);

    // Parse the column number.
    if (getToken().isNot(Token::integer)) {
      return emitWrongTokenError(
          "expected integer column number in FileLineColRange");
    }
    endColumn = getToken().getUnsignedIntegerValue();
    if (!endColumn.has_value())
      return emitError("expected integer column number in FileLineColRange");
    consumeToken(Token::integer);

    if (endLine.has_value()) {
      loc = FileLineColRange::get(StringAttr::get(ctx, str), *startLine,
                                  *startColumn, *endLine, *endColumn);
    } else {
      loc = FileLineColRange::get(StringAttr::get(ctx, str), *startLine,
                                  *startColumn, *endColumn);
    }
    return success();
  }

  // Otherwise, this is a NameLoc.

  // Check for a child location.
  if (consumeIf(Token::l_paren)) {
    // Parse the child location.
    LocationAttr childLoc;
    if (parseLocationInstance(childLoc))
      return failure();

    loc = NameLoc::get(StringAttr::get(ctx, str), childLoc);

    // Parse the closing ')'.
    if (parseToken(Token::r_paren,
                   "expected ')' after child location of NameLoc"))
      return failure();
  } else {
    loc = NameLoc::get(StringAttr::get(ctx, str));
  }

  return success();
}

ParseResult Parser::parseLocationInstance(LocationAttr &loc) {
  // Handle aliases.
  if (getToken().is(Token::hash_identifier)) {
    Attribute locAttr = parseExtendedAttr(Type());
    if (!locAttr)
      return failure();
    if (!(loc = dyn_cast<LocationAttr>(locAttr)))
      return emitError("expected location attribute, but got") << locAttr;
    return success();
  }

  // Handle either name or filelinecol locations.
  if (getToken().is(Token::string))
    return parseNameOrFileLineColRange(loc);

  // Bare tokens required for other cases.
  if (!getToken().is(Token::bare_identifier))
    return emitWrongTokenError("expected location instance");

  // Check for the 'callsite' signifying a callsite location.
  if (getToken().getSpelling() == "callsite")
    return parseCallSiteLocation(loc);

  // If the token is 'fused', then this is a fused location.
  if (getToken().getSpelling() == "fused")
    return parseFusedLocation(loc);

  // Check for a 'unknown' for an unknown location.
  if (getToken().getSpelling() == "unknown") {
    consumeToken(Token::bare_identifier);
    loc = UnknownLoc::get(getContext());
    return success();
  }

  return emitWrongTokenError("expected location instance");
}
