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
  /// A map from attribute alias identifier to Attribute.
  llvm::StringMap<Attribute> attributeAliasDefinitions;

  /// A map from type alias identifier to Type.
  llvm::StringMap<Type> typeAliasDefinitions;

  /// A map of dialect resource keys to the resolved resource name and handle
  /// to use during parsing.
  DenseMap<const OpAsmDialectInterface *,
           llvm::StringMap<std::pair<std::string, AsmDialectResourceHandle>>>
      dialectResources;
};

//===----------------------------------------------------------------------===//
// ParserState
//===----------------------------------------------------------------------===//

/// This class refers to all of the state maintained globally by the parser,
/// such as the current lexer position etc.
struct ParserState {
  ParserState(const llvm::SourceMgr &sourceMgr, const ParserConfig &config,
              SymbolState &symbols, AsmParserState *asmState,
              AsmParserCodeCompleteContext *codeCompleteContext)
      : config(config),
        lex(sourceMgr, config.getContext(), codeCompleteContext),
        curToken(lex.lexToken()), symbols(symbols), asmState(asmState),
        codeCompleteContext(codeCompleteContext) {}
  ParserState(const ParserState &) = delete;
  void operator=(const ParserState &) = delete;

  /// The configuration used to setup the parser.
  const ParserConfig &config;

  /// The lexer for the source file we're parsing.
  Lexer lex;

  /// This is the next token that hasn't been consumed yet.
  Token curToken;

  /// The current state for symbol parsing.
  SymbolState &symbols;

  /// An optional pointer to a struct containing high level parser state to be
  /// populated during parsing.
  AsmParserState *asmState;

  /// An optional code completion context.
  AsmParserCodeCompleteContext *codeCompleteContext;

  // Contains the stack of default dialect to use when parsing regions.
  // A new dialect get pushed to the stack before parsing regions nested
  // under an operation implementing `OpAsmOpInterface`, and
  // popped when done. At the top-level we start with "builtin" as the
  // default, so that the top-level `module` operation parses as-is.
  SmallVector<StringRef> defaultDialectStack{"builtin"};
};

} // namespace detail
} // namespace mlir

#endif // MLIR_LIB_PARSER_PARSERSTATE_H
