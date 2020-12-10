//===- LocationParser.cpp - MLIR Location Parser  -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Parser.h"

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
    return emitError("expected 'at' in callsite location");
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
      return emitError("expected valid attribute metadata");
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

  if (parseToken(Token::l_square, "expected '[' in fused location") ||
      parseCommaSeparatedList(parseElt) ||
      parseToken(Token::r_square, "expected ']' in fused location"))
    return failure();

  // Return the fused location.
  loc = FusedLoc::get(locations, metadata, getContext());
  return success();
}

ParseResult Parser::parseNameOrFileLineColLocation(LocationAttr &loc) {
  auto *ctx = getContext();
  auto str = getToken().getStringValue();
  consumeToken(Token::string);

  // If the next token is ':' this is a filelinecol location.
  if (consumeIf(Token::colon)) {
    // Parse the line number.
    if (getToken().isNot(Token::integer))
      return emitError("expected integer line number in FileLineColLoc");
    auto line = getToken().getUnsignedIntegerValue();
    if (!line.hasValue())
      return emitError("expected integer line number in FileLineColLoc");
    consumeToken(Token::integer);

    // Parse the ':'.
    if (parseToken(Token::colon, "expected ':' in FileLineColLoc"))
      return failure();

    // Parse the column number.
    if (getToken().isNot(Token::integer))
      return emitError("expected integer column number in FileLineColLoc");
    auto column = getToken().getUnsignedIntegerValue();
    if (!column.hasValue())
      return emitError("expected integer column number in FileLineColLoc");
    consumeToken(Token::integer);

    loc = FileLineColLoc::get(str, line.getValue(), column.getValue(), ctx);
    return success();
  }

  // Otherwise, this is a NameLoc.

  // Check for a child location.
  if (consumeIf(Token::l_paren)) {
    auto childSourceLoc = getToken().getLoc();

    // Parse the child location.
    LocationAttr childLoc;
    if (parseLocationInstance(childLoc))
      return failure();

    // The child must not be another NameLoc.
    if (childLoc.isa<NameLoc>())
      return emitError(childSourceLoc,
                       "child of NameLoc cannot be another NameLoc");
    loc = NameLoc::get(Identifier::get(str, ctx), childLoc);

    // Parse the closing ')'.
    if (parseToken(Token::r_paren,
                   "expected ')' after child location of NameLoc"))
      return failure();
  } else {
    loc = NameLoc::get(Identifier::get(str, ctx), ctx);
  }

  return success();
}

ParseResult Parser::parseLocationInstance(LocationAttr &loc) {
  // Handle either name or filelinecol locations.
  if (getToken().is(Token::string))
    return parseNameOrFileLineColLocation(loc);

  // Bare tokens required for other cases.
  if (!getToken().is(Token::bare_identifier))
    return emitError("expected location instance");

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

  return emitError("expected location instance");
}
