//===- ParserImpl.cpp - Presburger Parser Implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ParserImpl class for the Presburger textual form.
//
//===----------------------------------------------------------------------===//

#include "ParserImpl.h"
#include "Flattener.h"
#include "ParseStructs.h"
#include "mlir/Analysis/Presburger/Parser.h"

using namespace mlir;
using namespace presburger;
using llvm::MemoryBuffer;
using llvm::SourceMgr;

static bool isIdentifier(const Token &token) {
  return token.is(Token::bare_identifier) || token.isKeyword();
}

bool ParserImpl::parseToken(Token::Kind expectedToken, const Twine &message) {
  if (consumeIf(expectedToken))
    return true;
  return emitError(message);
}

bool ParserImpl::parseCommaSepeatedList(Delimiter delimiter,
                                        function_ref<bool()> parseElementFn,
                                        StringRef contextMessage) {
  switch (delimiter) {
  case Delimiter::None:
    break;
  case Delimiter::OptionalParen:
    if (getToken().isNot(Token::l_paren))
      return true;
    [[fallthrough]];
  case Delimiter::Paren:
    if (!parseToken(Token::l_paren, "expected '('" + contextMessage))
      return false;
    if (consumeIf(Token::r_paren))
      return true;
    break;
  case Delimiter::OptionalLessGreater:
    if (getToken().isNot(Token::less))
      return true;
    [[fallthrough]];
  case Delimiter::LessGreater:
    if (!parseToken(Token::less, "expected '<'" + contextMessage))
      return true;
    if (consumeIf(Token::greater))
      return true;
    break;
  case Delimiter::OptionalSquare:
    if (getToken().isNot(Token::l_square))
      return true;
    [[fallthrough]];
  case Delimiter::Square:
    if (!parseToken(Token::l_square, "expected '['" + contextMessage))
      return false;
    if (consumeIf(Token::r_square))
      return true;
    break;
  case Delimiter::OptionalBraces:
    if (getToken().isNot(Token::l_brace))
      return true;
    [[fallthrough]];
  case Delimiter::Braces:
    if (!parseToken(Token::l_brace, "expected '{'" + contextMessage))
      return false;
    if (consumeIf(Token::r_brace))
      return true;
    break;
  }

  if (!parseElementFn())
    return false;

  while (consumeIf(Token::comma))
    if (!parseElementFn())
      return false;

  switch (delimiter) {
  case Delimiter::None:
    return true;
  case Delimiter::OptionalParen:
  case Delimiter::Paren:
    return parseToken(Token::r_paren, "expected ')'" + contextMessage);
  case Delimiter::OptionalLessGreater:
  case Delimiter::LessGreater:
    return parseToken(Token::greater, "expected '>'" + contextMessage);
  case Delimiter::OptionalSquare:
  case Delimiter::Square:
    return parseToken(Token::r_square, "expected ']'" + contextMessage);
  case Delimiter::OptionalBraces:
  case Delimiter::Braces:
    return parseToken(Token::r_brace, "expected '}'" + contextMessage);
  }
  llvm_unreachable("Unknown delimiter");
}

bool ParserImpl::emitError(SMLoc loc, const Twine &message) {
  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().isNot(Token::error))
    sourceMgr.PrintMessage(loc, SourceMgr::DK_Error, message);
  return false;
}

bool ParserImpl::emitError(const Twine &message) {
  SMLoc loc = state.curToken.getLoc();
  if (state.curToken.isNot(Token::eof))
    return emitError(loc, message);

  // If the error is to be emitted at EOF, move it back one character.
  return emitError(SMLoc::getFromPointer(loc.getPointer() - 1), message);
}

/// Parse a bare id that may appear in an affine expression.
///
///   affine-expr ::= bare-id
PureAffineExpr ParserImpl::parseBareIdExpr() {
  if (!isIdentifier(getToken()))
    return emitError("expected bare identifier"), nullptr;

  StringRef sRef = getTokenSpelling();
  for (const auto &entry : dimsAndSymbols) {
    if (entry.first == sRef) {
      consumeToken();
      return std::make_unique<PureAffineExprImpl>(space, entry.second);
    }
  }

  return emitError("use of undeclared identifier"), nullptr;
}

/// Parse an affine expression inside parentheses.
///
///   affine-expr ::= `(` affine-expr `)`
PureAffineExpr ParserImpl::parseParentheticalExpr() {
  if (!parseToken(Token::l_paren, "expected '('"))
    return nullptr;
  if (getToken().is(Token::r_paren)) {
    return emitError("no expression inside parentheses"), nullptr;
  }

  PureAffineExpr expr = parseAffineExpr();
  if (!expr || !parseToken(Token::r_paren, "expected ')'"))
    return nullptr;

  return expr;
}

/// Parse the negation expression.
///
///   affine-expr ::= `-` affine-expr
PureAffineExpr ParserImpl::parseNegateExpression(const PureAffineExpr &lhs) {
  if (!parseToken(Token::minus, "expected '-'"))
    return nullptr;

  PureAffineExpr operand = parseAffineOperandExpr(lhs);
  // Since negation has the highest precedence of all ops (including high
  // precedence ops) but lower than parentheses, we are only going to use
  // parseAffineOperandExpr instead of parseAffineExpr here.
  if (!operand) {
    // Extra error message although parseAffineOperandExpr would have
    // complained. Leads to a better diagnostic.
    return emitError("missing operand of negation"), nullptr;
  }
  return -1 * std::move(operand);
}

/// Parse a positive integral constant appearing in an affine expression.
///
///   affine-expr ::= integer-literal
PureAffineExpr ParserImpl::parseIntegerExpr() {
  std::optional<uint64_t> val = getToken().getUInt64IntegerValue();
  if (!val)
    return emitError("failed to parse constant"), nullptr;
  int64_t ret = static_cast<int64_t>(*val);
  if (ret < 0)
    return emitError("constant too large"), nullptr;

  consumeToken(Token::integer);
  return std::make_unique<PureAffineExprImpl>(space, ret);
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
PureAffineExpr ParserImpl::parseAffineOperandExpr(const PureAffineExpr &lhs) {
  switch (getToken().getKind()) {
  case Token::integer:
    return parseIntegerExpr();
  case Token::l_paren:
    return parseParentheticalExpr();
  case Token::minus:
    return parseNegateExpression(lhs);
  case Token::kw_ceildiv:
  case Token::kw_floordiv:
  case Token::kw_mod:
    return parseBareIdExpr();
  case Token::plus:
  case Token::star:
    if (lhs)
      emitError("missing right operand of binary operator");
    else
      emitError("missing left operand of binary operator");
    return nullptr;
  default:
    if (isIdentifier(getToken()))
      return parseBareIdExpr();

    if (lhs)
      emitError("missing right operand of binary operator");
    else
      emitError("expected affine expression");
    return nullptr;
  }
}

PureAffineExpr ParserImpl::getAffineBinaryOpExpr(AffineHighPrecOp op,
                                                 PureAffineExpr &&lhs,
                                                 PureAffineExpr &&rhs,
                                                 SMLoc opLoc) {
  switch (op) {
  case Mul:
    if (!lhs->isConstant() && !rhs->isConstant()) {
      return emitError(opLoc,
                       "non-affine expression: at least one of the multiply "
                       "operands has to be a constant"),
             nullptr;
    }
    if (rhs->isConstant())
      return std::move(lhs) * rhs->getConstant();
    return std::move(rhs) * lhs->getConstant();
  case FloorDiv:
    if (!rhs->isConstant()) {
      return emitError(opLoc,
                       "non-affine expression: right operand of floordiv "
                       "has to be a constant"),
             nullptr;
    }
    return floordiv(std::move(lhs), rhs->getConstant());
  case CeilDiv:
    if (!rhs->isConstant()) {
      return emitError(opLoc, "non-affine expression: right operand of ceildiv "
                              "has to be a constant"),
             nullptr;
    }
    return ceildiv(std::move(lhs), rhs->getConstant());
  case Mod:
    if (!rhs->isConstant()) {
      return emitError(opLoc, "non-affine expression: right operand of mod "
                              "has to be a constant"),
             nullptr;
    }
    return std::move(lhs) % rhs->getConstant();
  case HNoOp:
    llvm_unreachable("can't create affine expression for null high prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown AffineHighPrecOp");
}

PureAffineExpr ParserImpl::getAffineBinaryOpExpr(AffineLowPrecOp op,
                                                 PureAffineExpr &&lhs,
                                                 PureAffineExpr &&rhs) {
  switch (op) {
  case AffineLowPrecOp::Add:
    return std::move(lhs) + std::move(rhs);
  case AffineLowPrecOp::Sub:
    return std::move(lhs) - std::move(rhs);
  case AffineLowPrecOp::LNoOp:
    llvm_unreachable("can't create affine expression for null low prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown AffineLowPrecOp");
}

AffineLowPrecOp ParserImpl::consumeIfLowPrecOp() {
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
AffineHighPrecOp ParserImpl::consumeIfHighPrecOp() {
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
PureAffineExpr ParserImpl::parseAffineHighPrecOpExpr(PureAffineExpr &&llhs,
                                                     AffineHighPrecOp llhsOp,
                                                     SMLoc llhsOpLoc) {
  PureAffineExpr lhs = parseAffineOperandExpr(llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Parse the remaining expression.
  SMLoc opLoc = getToken().getLoc();
  if (AffineHighPrecOp op = consumeIfHighPrecOp()) {
    if (llhs) {
      PureAffineExpr expr =
          getAffineBinaryOpExpr(llhsOp, std::move(llhs), std::move(lhs), opLoc);
      if (!expr)
        return nullptr;
      return parseAffineHighPrecOpExpr(std::move(expr), op, opLoc);
    }
    // No LLHS, get RHS
    return parseAffineHighPrecOpExpr(std::move(lhs), op, opLoc);
  }

  // This is the last operand in this expression.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, std::move(llhs), std::move(lhs),
                                 llhsOpLoc);

  return lhs;
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
PureAffineExpr ParserImpl::parseAffineLowPrecOpExpr(PureAffineExpr &&llhs,
                                                    AffineLowPrecOp llhsOp) {
  PureAffineExpr lhs = parseAffineOperandExpr(llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Deal with the ops.
  if (AffineLowPrecOp lOp = consumeIfLowPrecOp()) {
    if (llhs) {
      PureAffineExpr sum =
          getAffineBinaryOpExpr(llhsOp, std::move(llhs), std::move(lhs));
      return parseAffineLowPrecOpExpr(std::move(sum), lOp);
    }

    return parseAffineLowPrecOpExpr(std::move(lhs), lOp);
  }
  SMLoc opLoc = getToken().getLoc();
  if (AffineHighPrecOp hOp = consumeIfHighPrecOp()) {
    // We have a higher precedence op here. Get the rhs operand for the llhs
    // through parseAffineHighPrecOpExpr.
    PureAffineExpr highRes =
        parseAffineHighPrecOpExpr(std::move(lhs), hOp, opLoc);
    if (!highRes)
      return nullptr;

    // If llhs is null, the product forms the first operand of the yet to be
    // found expression. If non-null, the op to associate with llhs is llhsOp.
    PureAffineExpr expr = llhs ? getAffineBinaryOpExpr(llhsOp, std::move(llhs),
                                                       std::move(highRes))
                               : std::move(highRes);

    // Recurse for subsequent low prec op's after the affine high prec op
    // expression.
    if (AffineLowPrecOp nextOp = consumeIfLowPrecOp())
      return parseAffineLowPrecOpExpr(std::move(expr), nextOp);
    return expr;
  }
  // Last operand in the expression list.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, std::move(llhs), std::move(lhs));

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
PureAffineExpr ParserImpl::parseAffineExpr() {
  return parseAffineLowPrecOpExpr(nullptr, AffineLowPrecOp::LNoOp);
}

/// Parse a dim or symbol from the lists appearing before the actual
/// expressions of the affine map. Update our state to store the
/// dimensional/symbolic identifier.
bool ParserImpl::parseIdentifierDefinition(DimOrSymbolExpr idExpr) {
  if (!isIdentifier(getToken()))
    return emitError("expected bare identifier");

  StringRef name = getTokenSpelling();
  for (const auto &entry : dimsAndSymbols) {
    if (entry.first == name)
      return emitError("redefinition of identifier '" + name + "'");
  }
  consumeToken();

  dimsAndSymbols.emplace_back(name, idExpr);
  return true;
}

/// Parse the list of dimensional identifiers to an affine map.
bool ParserImpl::parseDimIdList() {
  auto parseElt = [&]() -> bool {
    return parseIdentifierDefinition(
        {DimOrSymbolKind::DimId, space.numSetDimVars()++});
  };
  return parseCommaSepeatedList(Delimiter::Paren, parseElt,
                                " in dimensional identifier list");
}

/// Parse the list of symbolic identifiers to an affine map.
bool ParserImpl::parseSymbolIdList() {
  auto parseElt = [&]() -> bool {
    return parseIdentifierDefinition(
        {DimOrSymbolKind::Symbol, space.numSymbolVars()++});
  };
  return parseCommaSepeatedList(Delimiter::Square, parseElt, " in symbol list");
}

/// Parse the list of symbolic identifiers to an affine map.
bool ParserImpl::parseDimAndOptionalSymbolIdList() {
  if (!parseDimIdList())
    return false;
  if (getToken().isNot(Token::l_square)) {
    space.numSymbolVars() = 0;
    return true;
  }
  return parseSymbolIdList();
}

/// Parse an affine constraint.
///  affine-constraint ::= affine-expr `>=` `affine-expr`
///                      | affine-expr `<=` `affine-expr`
///                      | affine-expr `==` `affine-expr`
///
/// The constraint is normalized to
///  affine-constraint ::= affine-expr `>=` `0`
///                      | affine-expr `==` `0`
/// before returning.
///
/// isEq is set to true if the parsed constraint is an equality, false if it
/// is an inequality (greater than or equal).
PureAffineExpr ParserImpl::parseAffineConstraint(bool *isEq) {
  PureAffineExpr lhsExpr = parseAffineExpr();
  if (!lhsExpr)
    return nullptr;

  if (consumeIf(Token::greater) && consumeIf(Token::equal)) {
    PureAffineExpr rhsExpr = parseAffineExpr();
    if (!rhsExpr)
      return nullptr;
    *isEq = false;
    return std::move(lhsExpr) - std::move(rhsExpr);
  }

  if (consumeIf(Token::less) && consumeIf(Token::equal)) {
    PureAffineExpr rhsExpr = parseAffineExpr();
    if (!rhsExpr)
      return nullptr;
    *isEq = false;
    return std::move(rhsExpr) - std::move(lhsExpr);
  }

  if (consumeIf(Token::equal) && consumeIf(Token::equal)) {
    PureAffineExpr rhsExpr = parseAffineExpr();
    if (!rhsExpr)
      return nullptr;
    *isEq = true;
    return std::move(lhsExpr) - std::move(rhsExpr);
  }

  return emitError("expected '==', '<=' or '>='"), nullptr;
}

FinalParseResult ParserImpl::parseAffineMapOrIntegerSet() {
  SmallVector<PureAffineExpr, 8> exprs;
  SmallVector<bool, 8> eqFlags;

  if (!parseDimAndOptionalSymbolIdList())
    llvm_unreachable("expected dim and symbol list");

  auto parseExpr = [&]() -> bool {
    PureAffineExpr elt = canonicalize(parseAffineExpr());
    if (elt) {
      exprs.emplace_back(std::move(elt));
      return true;
    }
    return false;
  };

  auto parseConstraint = [&]() -> bool {
    bool isEq;
    PureAffineExpr elt = canonicalize(parseAffineConstraint(&isEq));
    if (elt) {
      exprs.emplace_back(std::move(elt));
      eqFlags.push_back(isEq);
      return true;
    }
    return false;
  };

  if (consumeIf(Token::arrow)) {
    /// Parse the range and sizes affine map definition inline.
    ///
    ///  affine-map ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
    ///
    ///  multi-dim-affine-expr ::= `(` `)`
    ///  multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)`
    if (!parseCommaSepeatedList(Delimiter::Paren, parseExpr,
                                " in affine map range"))
      llvm_unreachable("expected affine map range");

    return {space, std::move(exprs), eqFlags};
  }

  if (!parseToken(Token::colon, "expected '->' or ':'"))
    llvm_unreachable("Unexpected token");

  /// Parse the constraints that are part of an integer set definition.
  ///
  ///  integer-set ::= dim-and-symbol-id-lists `:`
  ///                  '(' affine-constraint-conjunction? ')'
  ///  affine-constraint-conjunction ::= affine-constraint (`,`
  ///                                       affine-constraint)*
  ///
  if (!parseCommaSepeatedList(Delimiter::Paren, parseConstraint,
                              " in integer set constraint list"))
    llvm_unreachable("expected integer set");

  return {space, std::move(exprs), eqFlags};
}

static MultiAffineFunction getMAF(FinalParseResult &&parseResult) {
  auto space = parseResult.space;
  space.numDomainVars() = space.getNumSetDimVars();
  space.numRangeVars() = parseResult.exprs.size();

  auto [flatMatrix, cst] = Flattener(std::move(parseResult)).flatten();

  DivisionRepr divs = cst.getLocalReprs();
  assert(divs.hasAllReprs() &&
         "AffineMap cannot produce divs without local representation");

  return MultiAffineFunction(space, flatMatrix, divs);
}

static IntegerPolyhedron getPoly(FinalParseResult &&parseResult) {
  auto eqFlags = parseResult.eqFlags;
  auto [flatMatrix, cst] = Flattener(std::move(parseResult)).flatten();

  for (unsigned i = 0; i < flatMatrix.getNumRows(); ++i) {
    if (eqFlags[i])
      cst.addEquality(flatMatrix.getRow(i));
    else
      cst.addInequality(flatMatrix.getRow(i));
  }

  return cst;
}

static FinalParseResult parseAffineMapOrIntegerSet(StringRef str) {
  SourceMgr sourceMgr;
  auto memBuffer =
      MemoryBuffer::getMemBuffer(str, "<mlir_parser_buffer>", false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  ParserImpl parser(sourceMgr);
  return parser.parseAffineMapOrIntegerSet();
}

IntegerPolyhedron mlir::presburger::parseIntegerPolyhedron(StringRef str) {
  return getPoly(parseAffineMapOrIntegerSet(str));
}

MultiAffineFunction mlir::presburger::parseMultiAffineFunction(StringRef str) {
  return getMAF(parseAffineMapOrIntegerSet(str));
}
