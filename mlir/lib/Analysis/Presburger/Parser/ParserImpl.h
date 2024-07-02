//===- ParserImpl.h - Presburger Parser Implementation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PARSER_PARSERIMPL_H
#define MLIR_ANALYSIS_PRESBURGER_PARSER_PARSERIMPL_H

#include "Lexer.h"
#include "ParseStructs.h"

namespace mlir::presburger {
template <typename T>
using function_ref = llvm::function_ref<T>;
using llvm::SourceMgr;
using llvm::Twine;

/// This class refers to the lexing-related state for the parser.
struct ParserState {
  ParserState(const llvm::SourceMgr &sourceMgr)
      : lex(sourceMgr), curToken(lex.lexToken()), lastToken(Token::error, "") {}
  ParserState(const ParserState &) = delete;

  Lexer lex;
  Token curToken;
  Token lastToken;
};

/// These are the supported delimiters around operand lists and region
/// argument lists, used by parseOperandList.
enum class Delimiter {
  /// Zero or more operands with no delimiters.
  None,
  /// Parens surrounding zero or more operands.
  Paren,
  /// Square brackets surrounding zero or more operands.
  Square,
  /// <> brackets surrounding zero or more operands.
  LessGreater,
  /// {} brackets surrounding zero or more operands.
  Braces,
  /// Parens supporting zero or more operands, or nothing.
  OptionalParen,
  /// Square brackets supporting zero or more ops, or nothing.
  OptionalSquare,
  /// <> brackets supporting zero or more ops, or nothing.
  OptionalLessGreater,
  /// {} brackets surrounding zero or more operands, or nothing.
  OptionalBraces,
};

/// Lower precedence ops (all at the same precedence level). LNoOp is false in
/// the boolean sense.
enum AffineLowPrecOp {
  LNoOp, // Null value.
  Add,
  Sub
};

/// Higher precedence ops - all at the same precedence level. HNoOp is false
/// in the boolean sense.
enum AffineHighPrecOp {
  HNoOp, // Null value.
  Mul,
  FloorDiv,
  CeilDiv,
  Mod
};

class ParserImpl {
public:
  ParserImpl(const SourceMgr &sourceMgr)
      : sourceMgr(sourceMgr), state(sourceMgr) {}

  FinalParseResult parseAffineMapOrIntegerSet();

private:
  /// Parse a list of comma-separated items with an optional delimiter.  If a
  /// delimiter is provided, then an empty list is allowed.  If not, then at
  /// least one element will be parsed.
  bool parseCommaSepeatedList(Delimiter delimiter,
                              function_ref<bool()> parseElementFn,
                              StringRef contextMessage);

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//
  bool emitError(const Twine &message);
  bool emitError(SMLoc loc, const Twine &message);

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//
  const Token &getToken() const { return state.curToken; }
  StringRef getTokenSpelling() const { return state.curToken.getSpelling(); }
  const Token &getLastToken() const { return state.lastToken; }
  bool consumeIf(Token::Kind kind) {
    if (state.curToken.isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }
  void consumeToken() {
    assert(state.curToken.isNot(Token::eof, Token::error) &&
           "shouldn't advance past EOF or errors");
    state.lastToken = state.curToken;
    state.curToken = state.lex.lexToken();
  }
  void consumeToken(Token::Kind kind) {
    assert(state.curToken.is(kind) && "consumed an unexpected token");
    consumeToken();
  }
  void resetToken(const char *tokPos) {
    state.lex.resetPointer(tokPos);
    state.lastToken = state.curToken;
    state.curToken = state.lex.lexToken();
  }
  bool parseToken(Token::Kind expectedToken, const Twine &message);

  //===--------------------------------------------------------------------===//
  // Affine Parsing
  //===--------------------------------------------------------------------===//
  AffineLowPrecOp consumeIfLowPrecOp();
  AffineHighPrecOp consumeIfHighPrecOp();

  bool parseDimIdList();
  bool parseSymbolIdList();
  bool parseDimAndOptionalSymbolIdList();
  bool parseIdentifierDefinition(DimOrSymbolExpr idExpr);

  PureAffineExpr parseAffineExpr();
  PureAffineExpr parseParentheticalExpr();
  PureAffineExpr parseNegateExpression(const PureAffineExpr &lhs);
  PureAffineExpr parseIntegerExpr();
  PureAffineExpr parseBareIdExpr();

  PureAffineExpr getAffineBinaryOpExpr(AffineHighPrecOp op,
                                       PureAffineExpr &&lhs,
                                       PureAffineExpr &&rhs, SMLoc opLoc);
  PureAffineExpr getAffineBinaryOpExpr(AffineLowPrecOp op, PureAffineExpr &&lhs,
                                       PureAffineExpr &&rhs);
  PureAffineExpr parseAffineOperandExpr(const PureAffineExpr &lhs);
  PureAffineExpr parseAffineLowPrecOpExpr(PureAffineExpr &&llhs,
                                          AffineLowPrecOp llhsOp);
  PureAffineExpr parseAffineHighPrecOpExpr(PureAffineExpr &&llhs,
                                           AffineHighPrecOp llhsOp,
                                           SMLoc llhsOpLoc);
  PureAffineExpr parseAffineConstraint(bool *isEq);

  const SourceMgr &sourceMgr;
  ParserState state;
  ParseInfo info;
  SmallVector<std::pair<StringRef, DimOrSymbolExpr>, 4> dimsAndSymbols;
};
} // namespace mlir::presburger

#endif // MLIR_ANALYSIS_PRESBURGER_PARSER_PARSERIMPL_H
