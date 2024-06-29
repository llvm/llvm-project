//===--- Parser.h - Matcher expression parser -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple matcher expression parser.
//
// This file contains the Parser class, which is responsible for parsing
// expressions in a specific format: matcherName(Arg0, Arg1, ..., ArgN). The
// parser can also interpret simple types, like strings.
//
// The actual processing of the matchers is handled by a Sema object that is
// provided to the parser.
//
// The grammar for the supported expressions is as follows:
// <Expression>        := <StringLiteral> | <MatcherExpression>
// <StringLiteral>     := "quoted string"
// <MatcherExpression> := <MatcherName>(<ArgumentList>)
// <MatcherName>       := [a-zA-Z]+
// <ArgumentList>      := <Expression> | <Expression>,<ArgumentList>
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_PARSER_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_PARSER_H

#include "Diagnostics.h"
#include "RegistryManager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <vector>

namespace mlir::query::matcher::internal {

// Matcher expression parser.
class Parser {
public:
  // Different possible tokens.
  enum class TokenKind {
    Eof,
    NewLine,
    OpenParen,
    CloseParen,
    Comma,
    Period,
    Literal,
    Ident,
    InvalidChar,
    CodeCompletion,
    Error
  };

  // Interface to connect the parser with the registry and more. The parser uses
  // the Sema instance passed into parseMatcherExpression() to handle all
  // matcher tokens.
  class Sema {
  public:
    virtual ~Sema();

    // Process a matcher expression. The caller takes ownership of the Matcher
    // object returned.
    virtual VariantMatcher actOnMatcherExpression(
        MatcherCtor ctor, SourceRange nameRange, llvm::StringRef functionName,
        llvm::ArrayRef<ParserValue> args, Diagnostics *error) = 0;

    // Look up a matcher by name in the matcher name found by the parser.
    virtual std::optional<MatcherCtor>
    lookupMatcherCtor(llvm::StringRef matcherName) = 0;

    // Compute the list of completion types for Context.
    virtual std::vector<ArgKind> getAcceptedCompletionTypes(
        llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> Context);

    // Compute the list of completions that match any of acceptedTypes.
    virtual std::vector<MatcherCompletion>
    getMatcherCompletions(llvm::ArrayRef<ArgKind> acceptedTypes);
  };

  // An implementation of the Sema interface that uses the matcher registry to
  // process tokens.
  class RegistrySema : public Parser::Sema {
  public:
    RegistrySema(const Registry &matcherRegistry)
        : matcherRegistry(matcherRegistry) {}
    ~RegistrySema() override;

    std::optional<MatcherCtor>
    lookupMatcherCtor(llvm::StringRef matcherName) override;

    VariantMatcher actOnMatcherExpression(MatcherCtor Ctor,
                                          SourceRange NameRange,
                                          StringRef functionName,
                                          ArrayRef<ParserValue> Args,
                                          Diagnostics *Error) override;

    std::vector<ArgKind> getAcceptedCompletionTypes(
        llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> context) override;

    std::vector<MatcherCompletion>
    getMatcherCompletions(llvm::ArrayRef<ArgKind> acceptedTypes) override;

  private:
    const Registry &matcherRegistry;
  };

  using NamedValueMap = llvm::StringMap<VariantValue>;

  // Methods to parse a matcher expression and return a DynMatcher object,
  // transferring ownership to the caller.
  static std::optional<DynMatcher>
  parseMatcherExpression(llvm::StringRef &matcherCode,
                         const Registry &matcherRegistry,
                         const NamedValueMap *namedValues, Diagnostics *error);
  static std::optional<DynMatcher>
  parseMatcherExpression(llvm::StringRef &matcherCode,
                         const Registry &matcherRegistry, Diagnostics *error) {
    return parseMatcherExpression(matcherCode, matcherRegistry, nullptr, error);
  }

  // Methods to parse any expression supported by this parser.
  static bool parseExpression(llvm::StringRef &code,
                              const Registry &matcherRegistry,
                              const NamedValueMap *namedValues,
                              VariantValue *value, Diagnostics *error);

  static bool parseExpression(llvm::StringRef &code,
                              const Registry &matcherRegistry,
                              VariantValue *value, Diagnostics *error) {
    return parseExpression(code, matcherRegistry, nullptr, value, error);
  }

  // Methods to complete an expression at a given offset.
  static std::vector<MatcherCompletion>
  completeExpression(llvm::StringRef &code, unsigned completionOffset,
                     const Registry &matcherRegistry,
                     const NamedValueMap *namedValues);
  static std::vector<MatcherCompletion>
  completeExpression(llvm::StringRef &code, unsigned completionOffset,
                     const Registry &matcherRegistry) {
    return completeExpression(code, completionOffset, matcherRegistry, nullptr);
  }

private:
  class CodeTokenizer;
  struct ScopedContextEntry;
  struct TokenInfo;

  Parser(CodeTokenizer *tokenizer, const Registry &matcherRegistry,
         const NamedValueMap *namedValues, Diagnostics *error);

  bool parseChainedExpression(std::string &argument);

  bool parseExpressionImpl(VariantValue *value);

  bool parseMatcherArgs(std::vector<ParserValue> &args, MatcherCtor ctor,
                        const TokenInfo &nameToken, TokenInfo &endToken);

  bool parseMatcherExpressionImpl(const TokenInfo &nameToken,
                                  const TokenInfo &openToken,
                                  std::optional<MatcherCtor> ctor,
                                  VariantValue *value);

  bool parseIdentifierPrefixImpl(VariantValue *value);

  void addCompletion(const TokenInfo &compToken,
                     const MatcherCompletion &completion);
  void addExpressionCompletions();

  std::vector<MatcherCompletion>
  getNamedValueCompletions(llvm::ArrayRef<ArgKind> acceptedTypes);

  CodeTokenizer *const tokenizer;
  std::unique_ptr<RegistrySema> sema;
  const NamedValueMap *const namedValues;
  Diagnostics *const error;

  using ContextStackTy = std::vector<std::pair<MatcherCtor, unsigned>>;

  ContextStackTy contextStack;
  std::vector<MatcherCompletion> completions;
};

} // namespace mlir::query::matcher::internal

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_PARSER_H
