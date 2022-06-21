//===- AffineParser.cpp - MLIR Affine Parser ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a parser for Affine structures.
//
//===----------------------------------------------------------------------===//

#include "Parser.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::detail;

namespace {

/// Lower precedence ops (all at the same precedence level). LNoOp is false in
/// the boolean sense.
enum AffineLowPrecOp {
  /// Null value.
  LNoOp,
  Add,
  Sub
};

/// Higher precedence ops - all at the same precedence level. HNoOp is false
/// in the boolean sense.
enum AffineHighPrecOp {
  /// Null value.
  HNoOp,
  Mul,
  FloorDiv,
  CeilDiv,
  Mod
};

/// This is a specialized parser for affine structures (affine maps, affine
/// expressions, and integer sets), maintaining the state transient to their
/// bodies.
class AffineParser : public Parser {
public:
  AffineParser(ParserState &state, bool allowParsingSSAIds = false,
               function_ref<ParseResult(bool)> parseElement = nullptr)
      : Parser(state), allowParsingSSAIds(allowParsingSSAIds),
        parseElement(parseElement) {}

  ParseResult parseAffineMapRange(unsigned numDims, unsigned numSymbols,
                                  AffineMap &result);
  ParseResult parseAffineMapOrIntegerSetInline(AffineMap &map, IntegerSet &set);
  ParseResult parseIntegerSetConstraints(unsigned numDims, unsigned numSymbols,
                                         IntegerSet &result);
  ParseResult parseAffineMapOfSSAIds(AffineMap &map,
                                     OpAsmParser::Delimiter delimiter);
  ParseResult parseAffineExprOfSSAIds(AffineExpr &expr);
  void getDimsAndSymbolSSAIds(SmallVectorImpl<StringRef> &dimAndSymbolSSAIds,
                              unsigned &numDims);

private:
  // Binary affine op parsing.
  AffineLowPrecOp consumeIfLowPrecOp();
  AffineHighPrecOp consumeIfHighPrecOp();

  // Identifier lists for polyhedral structures.
  ParseResult parseDimIdList(unsigned &numDims);
  ParseResult parseSymbolIdList(unsigned &numSymbols);
  ParseResult parseDimAndOptionalSymbolIdList(unsigned &numDims,
                                              unsigned &numSymbols);
  ParseResult parseIdentifierDefinition(AffineExpr idExpr);

  AffineExpr parseAffineExpr();
  AffineExpr parseParentheticalExpr();
  AffineExpr parseNegateExpression(AffineExpr lhs);
  AffineExpr parseIntegerExpr();
  AffineExpr parseBareIdExpr();
  AffineExpr parseSSAIdExpr(bool isSymbol);
  AffineExpr parseSymbolSSAIdExpr();

  AffineExpr getAffineBinaryOpExpr(AffineHighPrecOp op, AffineExpr lhs,
                                   AffineExpr rhs, SMLoc opLoc);
  AffineExpr getAffineBinaryOpExpr(AffineLowPrecOp op, AffineExpr lhs,
                                   AffineExpr rhs);
  AffineExpr parseAffineOperandExpr(AffineExpr lhs);
  AffineExpr parseAffineLowPrecOpExpr(AffineExpr llhs, AffineLowPrecOp llhsOp);
  AffineExpr parseAffineHighPrecOpExpr(AffineExpr llhs, AffineHighPrecOp llhsOp,
                                       SMLoc llhsOpLoc);
  AffineExpr parseAffineConstraint(bool *isEq);

private:
  bool allowParsingSSAIds;
  function_ref<ParseResult(bool)> parseElement;
  unsigned numDimOperands = 0;
  unsigned numSymbolOperands = 0;
  SmallVector<std::pair<StringRef, AffineExpr>, 4> dimsAndSymbols;
};
} // namespace

/// Create an affine binary high precedence op expression (mul's, div's, mod).
/// opLoc is the location of the op token to be used to report errors
/// for non-conforming expressions.
AffineExpr AffineParser::getAffineBinaryOpExpr(AffineHighPrecOp op,
                                               AffineExpr lhs, AffineExpr rhs,
                                               SMLoc opLoc) {
  // TODO: make the error location info accurate.
  switch (op) {
  case Mul:
    if (!lhs.isSymbolicOrConstant() && !rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-affine expression: at least one of the multiply "
                       "operands has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs * rhs;
  case FloorDiv:
    if (!rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-affine expression: right operand of floordiv "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.floorDiv(rhs);
  case CeilDiv:
    if (!rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-affine expression: right operand of ceildiv "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.ceilDiv(rhs);
  case Mod:
    if (!rhs.isSymbolicOrConstant()) {
      emitError(opLoc, "non-affine expression: right operand of mod "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs % rhs;
  case HNoOp:
    llvm_unreachable("can't create affine expression for null high prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown AffineHighPrecOp");
}

/// Create an affine binary low precedence op expression (add, sub).
AffineExpr AffineParser::getAffineBinaryOpExpr(AffineLowPrecOp op,
                                               AffineExpr lhs, AffineExpr rhs) {
  switch (op) {
  case AffineLowPrecOp::Add:
    return lhs + rhs;
  case AffineLowPrecOp::Sub:
    return lhs - rhs;
  case AffineLowPrecOp::LNoOp:
    llvm_unreachable("can't create affine expression for null low prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown AffineLowPrecOp");
}

/// Consume this token if it is a lower precedence affine op (there are only
/// two precedence levels).
AffineLowPrecOp AffineParser::consumeIfLowPrecOp() {
  switch (getToken().getKind()) {
  case Token::plus:
    consumeToken(Token::plus);
    return AffineLowPrecOp::Add;
  case Token::minus:
    consumeToken(Token::minus);
    return AffineLowPrecOp::Sub;
  default:
    return AffineLowPrecOp::LNoOp;
  }
}

/// Consume this token if it is a higher precedence affine op (there are only
/// two precedence levels)
AffineHighPrecOp AffineParser::consumeIfHighPrecOp() {
  switch (getToken().getKind()) {
  case Token::star:
    consumeToken(Token::star);
    return Mul;
  case Token::kw_floordiv:
    consumeToken(Token::kw_floordiv);
    return FloorDiv;
  case Token::kw_ceildiv:
    consumeToken(Token::kw_ceildiv);
    return CeilDiv;
  case Token::kw_mod:
    consumeToken(Token::kw_mod);
    return Mod;
  default:
    return HNoOp;
  }
}

/// Parse a high precedence op expression list: mul, div, and mod are high
/// precedence binary ops, i.e., parse a
///   expr_1 op_1 expr_2 op_2 ... expr_n
/// where op_1, op_2 are all a AffineHighPrecOp (mul, div, mod).
/// All affine binary ops are left associative.
/// Given llhs, returns (llhs llhsOp lhs) op rhs, or (lhs op rhs) if llhs is
/// null. If no rhs can be found, returns (llhs llhsOp lhs) or lhs if llhs is
/// null. llhsOpLoc is the location of the llhsOp token that will be used to
/// report an error for non-conforming expressions.
AffineExpr AffineParser::parseAffineHighPrecOpExpr(AffineExpr llhs,
                                                   AffineHighPrecOp llhsOp,
                                                   SMLoc llhsOpLoc) {
  AffineExpr lhs = parseAffineOperandExpr(llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Parse the remaining expression.
  auto opLoc = getToken().getLoc();
  if (AffineHighPrecOp op = consumeIfHighPrecOp()) {
    if (llhs) {
      AffineExpr expr = getAffineBinaryOpExpr(llhsOp, llhs, lhs, opLoc);
      if (!expr)
        return nullptr;
      return parseAffineHighPrecOpExpr(expr, op, opLoc);
    }
    // No LLHS, get RHS
    return parseAffineHighPrecOpExpr(lhs, op, opLoc);
  }

  // This is the last operand in this expression.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, llhs, lhs, llhsOpLoc);

  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse an affine expression inside parentheses.
///
///   affine-expr ::= `(` affine-expr `)`
AffineExpr AffineParser::parseParentheticalExpr() {
  if (parseToken(Token::l_paren, "expected '('"))
    return nullptr;
  if (getToken().is(Token::r_paren))
    return emitError("no expression inside parentheses"), nullptr;

  auto expr = parseAffineExpr();
  if (!expr || parseToken(Token::r_paren, "expected ')'"))
    return nullptr;

  return expr;
}

/// Parse the negation expression.
///
///   affine-expr ::= `-` affine-expr
AffineExpr AffineParser::parseNegateExpression(AffineExpr lhs) {
  if (parseToken(Token::minus, "expected '-'"))
    return nullptr;

  AffineExpr operand = parseAffineOperandExpr(lhs);
  // Since negation has the highest precedence of all ops (including high
  // precedence ops) but lower than parentheses, we are only going to use
  // parseAffineOperandExpr instead of parseAffineExpr here.
  if (!operand)
    // Extra error message although parseAffineOperandExpr would have
    // complained. Leads to a better diagnostic.
    return emitError("missing operand of negation"), nullptr;
  return (-1) * operand;
}

/// Parse a bare id that may appear in an affine expression.
///
///   affine-expr ::= bare-id
AffineExpr AffineParser::parseBareIdExpr() {
  if (getToken().isNot(Token::bare_identifier))
    return emitWrongTokenError("expected bare identifier"), nullptr;

  StringRef sRef = getTokenSpelling();
  for (auto entry : dimsAndSymbols) {
    if (entry.first == sRef) {
      consumeToken(Token::bare_identifier);
      return entry.second;
    }
  }

  return emitWrongTokenError("use of undeclared identifier"), nullptr;
}

/// Parse an SSA id which may appear in an affine expression.
AffineExpr AffineParser::parseSSAIdExpr(bool isSymbol) {
  if (!allowParsingSSAIds)
    return emitWrongTokenError("unexpected ssa identifier"), nullptr;
  if (getToken().isNot(Token::percent_identifier))
    return emitWrongTokenError("expected ssa identifier"), nullptr;
  auto name = getTokenSpelling();
  // Check if we already parsed this SSA id.
  for (auto entry : dimsAndSymbols) {
    if (entry.first == name) {
      consumeToken(Token::percent_identifier);
      return entry.second;
    }
  }
  // Parse the SSA id and add an AffineDim/SymbolExpr to represent it.
  if (parseElement(isSymbol))
    return nullptr;
  auto idExpr = isSymbol
                    ? getAffineSymbolExpr(numSymbolOperands++, getContext())
                    : getAffineDimExpr(numDimOperands++, getContext());
  dimsAndSymbols.push_back({name, idExpr});
  return idExpr;
}

AffineExpr AffineParser::parseSymbolSSAIdExpr() {
  if (parseToken(Token::kw_symbol, "expected symbol keyword") ||
      parseToken(Token::l_paren, "expected '(' at start of SSA symbol"))
    return nullptr;
  AffineExpr symbolExpr = parseSSAIdExpr(/*isSymbol=*/true);
  if (!symbolExpr)
    return nullptr;
  if (parseToken(Token::r_paren, "expected ')' at end of SSA symbol"))
    return nullptr;
  return symbolExpr;
}

/// Parse a positive integral constant appearing in an affine expression.
///
///   affine-expr ::= integer-literal
AffineExpr AffineParser::parseIntegerExpr() {
  auto val = getToken().getUInt64IntegerValue();
  if (!val.hasValue() || (int64_t)val.getValue() < 0)
    return emitError("constant too large for index"), nullptr;

  consumeToken(Token::integer);
  return builder.getAffineConstantExpr((int64_t)val.getValue());
}

/// Parses an expression that can be a valid operand of an affine expression.
/// lhs: if non-null, lhs is an affine expression that is the lhs of a binary
/// operator, the rhs of which is being parsed. This is used to determine
/// whether an error should be emitted for a missing right operand.
//  Eg: for an expression without parentheses (like i + j + k + l), each
//  of the four identifiers is an operand. For i + j*k + l, j*k is not an
//  operand expression, it's an op expression and will be parsed via
//  parseAffineHighPrecOpExpression(). However, for i + (j*k) + -l, (j*k) and
//  -l are valid operands that will be parsed by this function.
AffineExpr AffineParser::parseAffineOperandExpr(AffineExpr lhs) {
  switch (getToken().getKind()) {
  case Token::bare_identifier:
    return parseBareIdExpr();
  case Token::kw_symbol:
    return parseSymbolSSAIdExpr();
  case Token::percent_identifier:
    return parseSSAIdExpr(/*isSymbol=*/false);
  case Token::integer:
    return parseIntegerExpr();
  case Token::l_paren:
    return parseParentheticalExpr();
  case Token::minus:
    return parseNegateExpression(lhs);
  case Token::kw_ceildiv:
  case Token::kw_floordiv:
  case Token::kw_mod:
  case Token::plus:
  case Token::star:
    if (lhs)
      emitError("missing right operand of binary operator");
    else
      emitError("missing left operand of binary operator");
    return nullptr;
  default:
    if (lhs)
      emitError("missing right operand of binary operator");
    else
      emitError("expected affine expression");
    return nullptr;
  }
}

/// Parse affine expressions that are bare-id's, integer constants,
/// parenthetical affine expressions, and affine op expressions that are a
/// composition of those.
///
/// All binary op's associate from left to right.
///
/// {add, sub} have lower precedence than {mul, div, and mod}.
///
/// Add, sub'are themselves at the same precedence level. Mul, floordiv,
/// ceildiv, and mod are at the same higher precedence level. Negation has
/// higher precedence than any binary op.
///
/// llhs: the affine expression appearing on the left of the one being parsed.
/// This function will return ((llhs llhsOp lhs) op rhs) if llhs is non null,
/// and lhs op rhs otherwise; if there is no rhs, llhs llhsOp lhs is returned
/// if llhs is non-null; otherwise lhs is returned. This is to deal with left
/// associativity.
///
/// Eg: when the expression is e1 + e2*e3 + e4, with e1 as llhs, this function
/// will return the affine expr equivalent of (e1 + (e2*e3)) + e4, where
/// (e2*e3) will be parsed using parseAffineHighPrecOpExpr().
AffineExpr AffineParser::parseAffineLowPrecOpExpr(AffineExpr llhs,
                                                  AffineLowPrecOp llhsOp) {
  AffineExpr lhs;
  if (!(lhs = parseAffineOperandExpr(llhs)))
    return nullptr;

  // Found an LHS. Deal with the ops.
  if (AffineLowPrecOp lOp = consumeIfLowPrecOp()) {
    if (llhs) {
      AffineExpr sum = getAffineBinaryOpExpr(llhsOp, llhs, lhs);
      return parseAffineLowPrecOpExpr(sum, lOp);
    }
    // No LLHS, get RHS and form the expression.
    return parseAffineLowPrecOpExpr(lhs, lOp);
  }
  auto opLoc = getToken().getLoc();
  if (AffineHighPrecOp hOp = consumeIfHighPrecOp()) {
    // We have a higher precedence op here. Get the rhs operand for the llhs
    // through parseAffineHighPrecOpExpr.
    AffineExpr highRes = parseAffineHighPrecOpExpr(lhs, hOp, opLoc);
    if (!highRes)
      return nullptr;

    // If llhs is null, the product forms the first operand of the yet to be
    // found expression. If non-null, the op to associate with llhs is llhsOp.
    AffineExpr expr =
        llhs ? getAffineBinaryOpExpr(llhsOp, llhs, highRes) : highRes;

    // Recurse for subsequent low prec op's after the affine high prec op
    // expression.
    if (AffineLowPrecOp nextOp = consumeIfLowPrecOp())
      return parseAffineLowPrecOpExpr(expr, nextOp);
    return expr;
  }
  // Last operand in the expression list.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, llhs, lhs);
  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse an affine expression.
///  affine-expr ::= `(` affine-expr `)`
///                | `-` affine-expr
///                | affine-expr `+` affine-expr
///                | affine-expr `-` affine-expr
///                | affine-expr `*` affine-expr
///                | affine-expr `floordiv` affine-expr
///                | affine-expr `ceildiv` affine-expr
///                | affine-expr `mod` affine-expr
///                | bare-id
///                | integer-literal
///
/// Additional conditions are checked depending on the production. For eg.,
/// one of the operands for `*` has to be either constant/symbolic; the second
/// operand for floordiv, ceildiv, and mod has to be a positive integer.
AffineExpr AffineParser::parseAffineExpr() {
  return parseAffineLowPrecOpExpr(nullptr, AffineLowPrecOp::LNoOp);
}

/// Parse a dim or symbol from the lists appearing before the actual
/// expressions of the affine map. Update our state to store the
/// dimensional/symbolic identifier.
ParseResult AffineParser::parseIdentifierDefinition(AffineExpr idExpr) {
  if (getToken().isNot(Token::bare_identifier))
    return emitWrongTokenError("expected bare identifier");

  auto name = getTokenSpelling();
  for (auto entry : dimsAndSymbols) {
    if (entry.first == name)
      return emitError("redefinition of identifier '" + name + "'");
  }
  consumeToken(Token::bare_identifier);

  dimsAndSymbols.push_back({name, idExpr});
  return success();
}

/// Parse the list of dimensional identifiers to an affine map.
ParseResult AffineParser::parseDimIdList(unsigned &numDims) {
  auto parseElt = [&]() -> ParseResult {
    auto dimension = getAffineDimExpr(numDims++, getContext());
    return parseIdentifierDefinition(dimension);
  };
  return parseCommaSeparatedList(Delimiter::Paren, parseElt,
                                 " in dimensional identifier list");
}

/// Parse the list of symbolic identifiers to an affine map.
ParseResult AffineParser::parseSymbolIdList(unsigned &numSymbols) {
  auto parseElt = [&]() -> ParseResult {
    auto symbol = getAffineSymbolExpr(numSymbols++, getContext());
    return parseIdentifierDefinition(symbol);
  };
  return parseCommaSeparatedList(Delimiter::Square, parseElt,
                                 " in symbol list");
}

/// Parse the list of symbolic identifiers to an affine map.
ParseResult
AffineParser::parseDimAndOptionalSymbolIdList(unsigned &numDims,
                                              unsigned &numSymbols) {
  if (parseDimIdList(numDims)) {
    return failure();
  }
  if (!getToken().is(Token::l_square)) {
    numSymbols = 0;
    return success();
  }
  return parseSymbolIdList(numSymbols);
}

/// Parses an ambiguous affine map or integer set definition inline.
ParseResult AffineParser::parseAffineMapOrIntegerSetInline(AffineMap &map,
                                                           IntegerSet &set) {
  unsigned numDims = 0, numSymbols = 0;

  // List of dimensional and optional symbol identifiers.
  if (parseDimAndOptionalSymbolIdList(numDims, numSymbols))
    return failure();

  if (consumeIf(Token::arrow))
    return parseAffineMapRange(numDims, numSymbols, map);

  if (parseToken(Token::colon, "expected '->' or ':'"))
    return failure();
  return parseIntegerSetConstraints(numDims, numSymbols, set);
}

/// Parse an AffineMap where the dim and symbol identifiers are SSA ids.
ParseResult
AffineParser::parseAffineMapOfSSAIds(AffineMap &map,
                                     OpAsmParser::Delimiter delimiter) {

  SmallVector<AffineExpr, 4> exprs;
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseAffineExpr();
    exprs.push_back(elt);
    return elt ? success() : failure();
  };

  // Parse a multi-dimensional affine expression (a comma-separated list of
  // 1-d affine expressions); the list can be empty. Grammar:
  // multi-dim-affine-expr ::= `(` `)`
  //                         | `(` affine-expr (`,` affine-expr)* `)`
  if (parseCommaSeparatedList(delimiter, parseElt, " in affine map"))
    return failure();

  // Parsed a valid affine map.
  map = AffineMap::get(numDimOperands, dimsAndSymbols.size() - numDimOperands,
                       exprs, getContext());
  return success();
}

/// Parse an AffineExpr where the dim and symbol identifiers are SSA ids.
ParseResult AffineParser::parseAffineExprOfSSAIds(AffineExpr &expr) {
  expr = parseAffineExpr();
  return success(expr != nullptr);
}

/// Parse the range and sizes affine map definition inline.
///
///  affine-map ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
///
///  multi-dim-affine-expr ::= `(` `)`
///  multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)`
ParseResult AffineParser::parseAffineMapRange(unsigned numDims,
                                              unsigned numSymbols,
                                              AffineMap &result) {
  SmallVector<AffineExpr, 4> exprs;
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseAffineExpr();
    ParseResult res = elt ? success() : failure();
    exprs.push_back(elt);
    return res;
  };

  // Parse a multi-dimensional affine expression (a comma-separated list of
  // 1-d affine expressions). Grammar:
  // multi-dim-affine-expr ::= `(` `)`
  //                         | `(` affine-expr (`,` affine-expr)* `)`
  if (parseCommaSeparatedList(Delimiter::Paren, parseElt,
                              " in affine map range"))
    return failure();

  // Parsed a valid affine map.
  result = AffineMap::get(numDims, numSymbols, exprs, getContext());
  return success();
}

/// Parse an affine constraint.
///  affine-constraint ::= affine-expr `>=` `0`
///                      | affine-expr `==` `0`
///
/// isEq is set to true if the parsed constraint is an equality, false if it
/// is an inequality (greater than or equal).
///
AffineExpr AffineParser::parseAffineConstraint(bool *isEq) {
  AffineExpr expr = parseAffineExpr();
  if (!expr)
    return nullptr;

  if (consumeIf(Token::greater) && consumeIf(Token::equal) &&
      getToken().is(Token::integer)) {
    auto dim = getToken().getUnsignedIntegerValue();
    if (dim && *dim == 0) {
      consumeToken(Token::integer);
      *isEq = false;
      return expr;
    }
    return emitError("expected '0' after '>='"), nullptr;
  }

  if (consumeIf(Token::equal) && consumeIf(Token::equal) &&
      getToken().is(Token::integer)) {
    auto dim = getToken().getUnsignedIntegerValue();
    if (dim && *dim == 0) {
      consumeToken(Token::integer);
      *isEq = true;
      return expr;
    }
    return emitError("expected '0' after '=='"), nullptr;
  }

  return emitError("expected '== 0' or '>= 0' at end of affine constraint"),
         nullptr;
}

/// Parse the constraints that are part of an integer set definition.
///  integer-set-inline
///                ::= dim-and-symbol-id-lists `:`
///                '(' affine-constraint-conjunction? ')'
///  affine-constraint-conjunction ::= affine-constraint (`,`
///                                       affine-constraint)*
///
ParseResult AffineParser::parseIntegerSetConstraints(unsigned numDims,
                                                     unsigned numSymbols,
                                                     IntegerSet &result) {
  SmallVector<AffineExpr, 4> constraints;
  SmallVector<bool, 4> isEqs;
  auto parseElt = [&]() -> ParseResult {
    bool isEq;
    auto elt = parseAffineConstraint(&isEq);
    ParseResult res = elt ? success() : failure();
    if (elt) {
      constraints.push_back(elt);
      isEqs.push_back(isEq);
    }
    return res;
  };

  // Parse a list of affine constraints (comma-separated).
  if (parseCommaSeparatedList(Delimiter::Paren, parseElt,
                              " in integer set constraint list"))
    return failure();

  // If no constraints were parsed, then treat this as a degenerate 'true' case.
  if (constraints.empty()) {
    /* 0 == 0 */
    auto zero = getAffineConstantExpr(0, getContext());
    result = IntegerSet::get(numDims, numSymbols, zero, true);
    return success();
  }

  // Parsed a valid integer set.
  result = IntegerSet::get(numDims, numSymbols, constraints, isEqs);
  return success();
}

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// Parse an ambiguous reference to either and affine map or an integer set.
ParseResult Parser::parseAffineMapOrIntegerSetReference(AffineMap &map,
                                                        IntegerSet &set) {
  return AffineParser(state).parseAffineMapOrIntegerSetInline(map, set);
}
ParseResult Parser::parseAffineMapReference(AffineMap &map) {
  SMLoc curLoc = getToken().getLoc();
  IntegerSet set;
  if (parseAffineMapOrIntegerSetReference(map, set))
    return failure();
  if (set)
    return emitError(curLoc, "expected AffineMap, but got IntegerSet");
  return success();
}
ParseResult Parser::parseIntegerSetReference(IntegerSet &set) {
  SMLoc curLoc = getToken().getLoc();
  AffineMap map;
  if (parseAffineMapOrIntegerSetReference(map, set))
    return failure();
  if (map)
    return emitError(curLoc, "expected IntegerSet, but got AffineMap");
  return success();
}

/// Parse an AffineMap of SSA ids. The callback 'parseElement' is used to
/// parse SSA value uses encountered while parsing affine expressions.
ParseResult
Parser::parseAffineMapOfSSAIds(AffineMap &map,
                               function_ref<ParseResult(bool)> parseElement,
                               OpAsmParser::Delimiter delimiter) {
  return AffineParser(state, /*allowParsingSSAIds=*/true, parseElement)
      .parseAffineMapOfSSAIds(map, delimiter);
}

/// Parse an AffineExpr of SSA ids. The callback `parseElement` is used to parse
/// SSA value uses encountered while parsing.
ParseResult
Parser::parseAffineExprOfSSAIds(AffineExpr &expr,
                                function_ref<ParseResult(bool)> parseElement) {
  return AffineParser(state, /*allowParsingSSAIds=*/true, parseElement)
      .parseAffineExprOfSSAIds(expr);
}

IntegerSet mlir::parseIntegerSet(StringRef inputStr, MLIRContext *context,
                                 bool printDiagnosticInfo) {
  llvm::SourceMgr sourceMgr;
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(
      inputStr, /*BufferName=*/"<mlir_parser_buffer>",
      /*RequiresNullTerminator=*/false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  SymbolState symbolState;
  ParserState state(sourceMgr, context, symbolState, /*asmState=*/nullptr);
  Parser parser(state);

  raw_ostream &os = printDiagnosticInfo ? llvm::errs() : llvm::nulls();
  SourceMgrDiagnosticHandler handler(sourceMgr, context, os);
  IntegerSet set;
  if (parser.parseIntegerSetReference(set))
    return IntegerSet();

  Token endTok = parser.getToken();
  if (endTok.isNot(Token::eof)) {
    parser.emitError(endTok.getLoc(), "encountered unexpected token");
    return IntegerSet();
  }

  return set;
}
