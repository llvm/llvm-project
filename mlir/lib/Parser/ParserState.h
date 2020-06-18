//===- ParserState.h - MLIR ParserState -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_PARSER_PARSERSTATE_H
#define MLIR_LIB_PARSER_PARSERSTATE_H

#include "Lexer.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace detail {

//===----------------------------------------------------------------------===//
// SymbolState
//===----------------------------------------------------------------------===//

/// This class contains record of any parsed top-level symbols.
struct SymbolState {
  // A map from attribute alias identifier to Attribute.
  llvm::StringMap<Attribute> attributeAliasDefinitions;

  // A map from type alias identifier to Type.
  llvm::StringMap<Type> typeAliasDefinitions;

  /// A set of locations into the main parser memory buffer for each of the
  /// active nested parsers. Given that some nested parsers, i.e. custom dialect
  /// parsers, operate on a temporary memory buffer, this provides an anchor
  /// point for emitting diagnostics.
  SmallVector<llvm::SMLoc, 1> nestedParserLocs;

  /// The top-level lexer that contains the original memory buffer provided by
  /// the user. This is used by nested parsers to get a properly encoded source
  /// location.
  Lexer *topLevelLexer = nullptr;
};

//===----------------------------------------------------------------------===//
// ParserState
//===----------------------------------------------------------------------===//

/// This class refers to all of the state maintained globally by the parser,
/// such as the current lexer position etc.
struct ParserState {
  ParserState(const llvm::SourceMgr &sourceMgr, MLIRContext *ctx,
              SymbolState &symbols)
      : context(ctx), lex(sourceMgr, ctx), curToken(lex.lexToken()),
        symbols(symbols), parserDepth(symbols.nestedParserLocs.size()) {
    // Set the top level lexer for the symbol state if one doesn't exist.
    if (!symbols.topLevelLexer)
      symbols.topLevelLexer = &lex;
  }
  ~ParserState() {
    // Reset the top level lexer if it refers the lexer in our state.
    if (symbols.topLevelLexer == &lex)
      symbols.topLevelLexer = nullptr;
  }
  ParserState(const ParserState &) = delete;
  void operator=(const ParserState &) = delete;

  /// The context we're parsing into.
  MLIRContext *const context;

  /// The lexer for the source file we're parsing.
  Lexer lex;

  /// This is the next token that hasn't been consumed yet.
  Token curToken;

  /// The current state for symbol parsing.
  SymbolState &symbols;

  /// The depth of this parser in the nested parsing stack.
  size_t parserDepth;
};

} // end namespace detail
} // end namespace mlir

#endif // MLIR_LIB_PARSER_PARSERSTATE_H
