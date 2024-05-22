//===- Parser.cpp - Matcher expression parser -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Recursive parser implementation for the matcher expression grammar.
//
//===----------------------------------------------------------------------===//

#include "Parser.h"

#include <vector>

namespace mlir::query::matcher::internal {

// Simple structure to hold information for one token from the parser.
struct Parser::TokenInfo {
  TokenInfo() = default;

  // Method to set the kind and text of the token
  void set(TokenKind newKind, llvm::StringRef newText) {
    kind = newKind;
    text = newText;
  }

  llvm::StringRef text;
  TokenKind kind = TokenKind::Eof;
  SourceRange range;
  VariantValue value;
};

class Parser::CodeTokenizer {
public:
  // Constructor with matcherCode and error
  explicit CodeTokenizer(llvm::StringRef matcherCode, Diagnostics *error)
      : code(matcherCode), startOfLine(matcherCode), error(error) {
    nextToken = getNextToken();
  }

  // Constructor with matcherCode, error, and codeCompletionOffset
  CodeTokenizer(llvm::StringRef matcherCode, Diagnostics *error,
                unsigned codeCompletionOffset)
      : code(matcherCode), startOfLine(matcherCode), error(error),
        codeCompletionLocation(matcherCode.data() + codeCompletionOffset) {
    nextToken = getNextToken();
  }

  // Peek at next token without consuming it
  const TokenInfo &peekNextToken() const { return nextToken; }

  // Consume and return the next token
  TokenInfo consumeNextToken() {
    TokenInfo thisToken = nextToken;
    nextToken = getNextToken();
    return thisToken;
  }

  // Skip any newline tokens
  TokenInfo skipNewlines() {
    while (nextToken.kind == TokenKind::NewLine)
      nextToken = getNextToken();
    return nextToken;
  }

  // Consume and return next token, ignoring newlines
  TokenInfo consumeNextTokenIgnoreNewlines() {
    skipNewlines();
    return nextToken.kind == TokenKind::Eof ? nextToken : consumeNextToken();
  }

  // Return kind of next token
  TokenKind nextTokenKind() const { return nextToken.kind; }

private:
  // Helper function to get the first character as a new StringRef and drop it
  // from the original string
  llvm::StringRef firstCharacterAndDrop(llvm::StringRef &str) {
    assert(!str.empty());
    llvm::StringRef firstChar = str.substr(0, 1);
    str = str.drop_front();
    return firstChar;
  }

  // Get next token, consuming whitespaces and handling different token types
  TokenInfo getNextToken() {
    consumeWhitespace();
    TokenInfo result;
    result.range.start = currentLocation();

    // Code completion case
    if (codeCompletionLocation && codeCompletionLocation <= code.data()) {
      result.set(TokenKind::CodeCompletion,
                 llvm::StringRef(codeCompletionLocation, 0));
      codeCompletionLocation = nullptr;
      return result;
    }

    // End of file case
    if (code.empty()) {
      result.set(TokenKind::Eof, "");
      return result;
    }

    // Switch to handle specific characters
    switch (code[0]) {
    case '#':
      code = code.drop_until([](char c) { return c == '\n'; });
      return getNextToken();
    case ',':
      result.set(TokenKind::Comma, firstCharacterAndDrop(code));
      break;
    case '.':
      result.set(TokenKind::Period, firstCharacterAndDrop(code));
      break;
    case '\n':
      ++line;
      startOfLine = code.drop_front();
      result.set(TokenKind::NewLine, firstCharacterAndDrop(code));
      break;
    case '(':
      result.set(TokenKind::OpenParen, firstCharacterAndDrop(code));
      break;
    case ')':
      result.set(TokenKind::CloseParen, firstCharacterAndDrop(code));
      break;
    case '"':
    case '\'':
      consumeStringLiteral(&result);
      break;
    default:
      parseIdentifierOrInvalid(&result);
      break;
    }

    result.range.end = currentLocation();
    return result;
  }

  // Consume a string literal, handle escape sequences and missing closing
  // quote.
  void consumeStringLiteral(TokenInfo *result) {
    bool inEscape = false;
    const char marker = code[0];
    for (size_t length = 1; length < code.size(); ++length) {
      if (inEscape) {
        inEscape = false;
        continue;
      }
      if (code[length] == '\\') {
        inEscape = true;
        continue;
      }
      if (code[length] == marker) {
        result->kind = TokenKind::Literal;
        result->text = code.substr(0, length + 1);
        result->value = code.substr(1, length - 1);
        code = code.drop_front(length + 1);
        return;
      }
    }
    llvm::StringRef errorText = code;
    code = code.drop_front(code.size());
    SourceRange range;
    range.start = result->range.start;
    range.end = currentLocation();
    error->addError(range, ErrorType::ParserStringError) << errorText;
    result->kind = TokenKind::Error;
  }

  void parseIdentifierOrInvalid(TokenInfo *result) {
    if (isalnum(code[0])) {
      // Parse an identifier
      size_t tokenLength = 1;

      while (true) {
        // A code completion location in/immediately after an identifier will
        // cause the portion of the identifier before the code completion
        // location to become a code completion token.
        if (codeCompletionLocation == code.data() + tokenLength) {
          codeCompletionLocation = nullptr;
          result->kind = TokenKind::CodeCompletion;
          result->text = code.substr(0, tokenLength);
          code = code.drop_front(tokenLength);
          return;
        }
        if (tokenLength == code.size() || !(isalnum(code[tokenLength])))
          break;
        ++tokenLength;
      }
      result->kind = TokenKind::Ident;
      result->text = code.substr(0, tokenLength);
      code = code.drop_front(tokenLength);
    } else {
      result->kind = TokenKind::InvalidChar;
      result->text = code.substr(0, 1);
      code = code.drop_front(1);
    }
  }

  // Consume all leading whitespace from code, except newlines
  void consumeWhitespace() {
    code = code.drop_while(
        [](char c) { return llvm::StringRef(" \t\v\f\r").contains(c); });
  }

  // Returns the current location in the source code
  SourceLocation currentLocation() {
    SourceLocation location;
    location.line = line;
    location.column = code.data() - startOfLine.data() + 1;
    return location;
  }

  llvm::StringRef code;
  llvm::StringRef startOfLine;
  unsigned line = 1;
  Diagnostics *error;
  TokenInfo nextToken;
  const char *codeCompletionLocation = nullptr;
};

Parser::Sema::~Sema() = default;

std::vector<ArgKind> Parser::Sema::getAcceptedCompletionTypes(
    llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> context) {
  return {};
}

std::vector<MatcherCompletion>
Parser::Sema::getMatcherCompletions(llvm::ArrayRef<ArgKind> acceptedTypes) {
  return {};
}

// Entry for the scope of a parser
struct Parser::ScopedContextEntry {
  Parser *parser;

  ScopedContextEntry(Parser *parser, MatcherCtor c) : parser(parser) {
    parser->contextStack.emplace_back(c, 0u);
  }

  ~ScopedContextEntry() { parser->contextStack.pop_back(); }

  void nextArg() { ++parser->contextStack.back().second; }
};

// Parse and validate expressions starting with an identifier.
// This function can parse named values and matchers. In case of failure, it
// will try to determine the user's intent to give an appropriate error message.
bool Parser::parseIdentifierPrefixImpl(VariantValue *value) {
  const TokenInfo nameToken = tokenizer->consumeNextToken();

  if (tokenizer->nextTokenKind() != TokenKind::OpenParen) {
    // Parse as a named value.
    auto namedValue =
        namedValues ? namedValues->lookup(nameToken.text) : VariantValue();

    if (!namedValue.isMatcher()) {
      error->addError(tokenizer->peekNextToken().range,
                      ErrorType::ParserNotAMatcher);
      return false;
    }

    if (tokenizer->nextTokenKind() == TokenKind::NewLine) {
      error->addError(tokenizer->peekNextToken().range,
                      ErrorType::ParserNoOpenParen)
          << "NewLine";
      return false;
    }

    // If the syntax is correct and the name is not a matcher either, report
    // an unknown named value.
    if ((tokenizer->nextTokenKind() == TokenKind::Comma ||
         tokenizer->nextTokenKind() == TokenKind::CloseParen ||
         tokenizer->nextTokenKind() == TokenKind::NewLine ||
         tokenizer->nextTokenKind() == TokenKind::Eof) &&
        !sema->lookupMatcherCtor(nameToken.text)) {
      error->addError(nameToken.range, ErrorType::RegistryValueNotFound)
          << nameToken.text;
      return false;
    }
    // Otherwise, fallback to the matcher parser.
  }

  tokenizer->skipNewlines();

  assert(nameToken.kind == TokenKind::Ident);
  TokenInfo openToken = tokenizer->consumeNextToken();
  if (openToken.kind != TokenKind::OpenParen) {
    error->addError(openToken.range, ErrorType::ParserNoOpenParen)
        << openToken.text;
    return false;
  }

  std::optional<MatcherCtor> ctor = sema->lookupMatcherCtor(nameToken.text);

  // Parse as a matcher expression.
  return parseMatcherExpressionImpl(nameToken, openToken, ctor, value);
}

// Parse the arguments of a matcher
bool Parser::parseMatcherArgs(std::vector<ParserValue> &args, MatcherCtor ctor,
                              const TokenInfo &nameToken, TokenInfo &endToken) {
  ScopedContextEntry sce(this, ctor);

  while (tokenizer->nextTokenKind() != TokenKind::Eof) {
    if (tokenizer->nextTokenKind() == TokenKind::CloseParen) {
      // end of args.
      endToken = tokenizer->consumeNextToken();
      break;
    }

    if (!args.empty()) {
      // We must find a , token to continue.
      TokenInfo commaToken = tokenizer->consumeNextToken();
      if (commaToken.kind != TokenKind::Comma) {
        error->addError(commaToken.range, ErrorType::ParserNoComma)
            << commaToken.text;
        return false;
      }
    }

    ParserValue argValue;
    tokenizer->skipNewlines();

    argValue.text = tokenizer->peekNextToken().text;
    argValue.range = tokenizer->peekNextToken().range;
    if (!parseExpressionImpl(&argValue.value)) {
      return false;
    }

    tokenizer->skipNewlines();
    args.push_back(argValue);
    sce.nextArg();
  }

  return true;
}

// Parse and validate a matcher expression.
bool Parser::parseMatcherExpressionImpl(const TokenInfo &nameToken,
                                        const TokenInfo &openToken,
                                        std::optional<MatcherCtor> ctor,
                                        VariantValue *value) {
  if (!ctor) {
    error->addError(nameToken.range, ErrorType::RegistryMatcherNotFound)
        << nameToken.text;
    // Do not return here. We need to continue to give completion suggestions.
  }

  std::vector<ParserValue> args;
  TokenInfo endToken;

  tokenizer->skipNewlines();

  if (!parseMatcherArgs(args, ctor.value_or(nullptr), nameToken, endToken)) {
    return false;
  }

  // Check for the missing closing parenthesis
  if (endToken.kind != TokenKind::CloseParen) {
    error->addError(openToken.range, ErrorType::ParserNoCloseParen)
        << nameToken.text;
    return false;
  }

  if (!ctor)
    return false;
  // Merge the start and end infos.
  SourceRange matcherRange = nameToken.range;
  matcherRange.end = endToken.range.end;
  VariantMatcher result =
      sema->actOnMatcherExpression(*ctor, matcherRange, args, error);
  if (result.isNull())
    return false;
  *value = result;
  return true;
}

// If the prefix of this completion matches the completion token, add it to
// completions minus the prefix.
void Parser::addCompletion(const TokenInfo &compToken,
                           const MatcherCompletion &completion) {
  if (llvm::StringRef(completion.typedText).starts_with(compToken.text)) {
    completions.emplace_back(completion.typedText.substr(compToken.text.size()),
                             completion.matcherDecl);
  }
}

std::vector<MatcherCompletion>
Parser::getNamedValueCompletions(llvm::ArrayRef<ArgKind> acceptedTypes) {
  if (!namedValues)
    return {};

  std::vector<MatcherCompletion> result;
  for (const auto &entry : *namedValues) {
    std::string decl =
        (entry.getValue().getTypeAsString() + " " + entry.getKey()).str();
    result.emplace_back(entry.getKey(), decl);
  }
  return result;
}

void Parser::addExpressionCompletions() {
  const TokenInfo compToken = tokenizer->consumeNextTokenIgnoreNewlines();
  assert(compToken.kind == TokenKind::CodeCompletion);

  // We cannot complete code if there is an invalid element on the context
  // stack.
  for (const auto &entry : contextStack) {
    if (!entry.first)
      return;
  }

  auto acceptedTypes = sema->getAcceptedCompletionTypes(contextStack);
  for (const auto &completion : sema->getMatcherCompletions(acceptedTypes)) {
    addCompletion(compToken, completion);
  }

  for (const auto &completion : getNamedValueCompletions(acceptedTypes)) {
    addCompletion(compToken, completion);
  }
}

// Parse an <Expresssion>
bool Parser::parseExpressionImpl(VariantValue *value) {
  switch (tokenizer->nextTokenKind()) {
  case TokenKind::Literal:
    *value = tokenizer->consumeNextToken().value;
    return true;
  case TokenKind::Ident:
    return parseIdentifierPrefixImpl(value);
  case TokenKind::CodeCompletion:
    addExpressionCompletions();
    return false;
  case TokenKind::Eof:
    error->addError(tokenizer->consumeNextToken().range,
                    ErrorType::ParserNoCode);
    return false;

  case TokenKind::Error:
    // This error was already reported by the tokenizer.
    return false;
  case TokenKind::NewLine:
  case TokenKind::OpenParen:
  case TokenKind::CloseParen:
  case TokenKind::Comma:
  case TokenKind::Period:
  case TokenKind::InvalidChar:
    const TokenInfo token = tokenizer->consumeNextToken();
    error->addError(token.range, ErrorType::ParserInvalidToken)
        << (token.kind == TokenKind::NewLine ? "NewLine" : token.text);
    return false;
  }

  llvm_unreachable("Unknown token kind.");
}

Parser::Parser(CodeTokenizer *tokenizer, const Registry &matcherRegistry,
               const NamedValueMap *namedValues, Diagnostics *error)
    : tokenizer(tokenizer),
      sema(std::make_unique<RegistrySema>(matcherRegistry)),
      namedValues(namedValues), error(error) {}

Parser::RegistrySema::~RegistrySema() = default;

std::optional<MatcherCtor>
Parser::RegistrySema::lookupMatcherCtor(llvm::StringRef matcherName) {
  return RegistryManager::lookupMatcherCtor(matcherName, matcherRegistry);
}

VariantMatcher Parser::RegistrySema::actOnMatcherExpression(
    MatcherCtor ctor, SourceRange nameRange, llvm::ArrayRef<ParserValue> args,
    Diagnostics *error) {
  return RegistryManager::constructMatcher(ctor, nameRange, args, error);
}

std::vector<ArgKind> Parser::RegistrySema::getAcceptedCompletionTypes(
    llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> context) {
  return RegistryManager::getAcceptedCompletionTypes(context);
}

std::vector<MatcherCompletion> Parser::RegistrySema::getMatcherCompletions(
    llvm::ArrayRef<ArgKind> acceptedTypes) {
  return RegistryManager::getMatcherCompletions(acceptedTypes, matcherRegistry);
}

bool Parser::parseExpression(llvm::StringRef &code,
                             const Registry &matcherRegistry,
                             const NamedValueMap *namedValues,
                             VariantValue *value, Diagnostics *error) {
  CodeTokenizer tokenizer(code, error);
  Parser parser(&tokenizer, matcherRegistry, namedValues, error);
  if (!parser.parseExpressionImpl(value))
    return false;
  auto nextToken = tokenizer.peekNextToken();
  if (nextToken.kind != TokenKind::Eof &&
      nextToken.kind != TokenKind::NewLine) {
    error->addError(tokenizer.peekNextToken().range,
                    ErrorType::ParserTrailingCode);
    return false;
  }
  return true;
}

std::vector<MatcherCompletion>
Parser::completeExpression(llvm::StringRef &code, unsigned completionOffset,
                           const Registry &matcherRegistry,
                           const NamedValueMap *namedValues) {
  Diagnostics error;
  CodeTokenizer tokenizer(code, &error, completionOffset);
  Parser parser(&tokenizer, matcherRegistry, namedValues, &error);
  VariantValue dummy;
  parser.parseExpressionImpl(&dummy);

  return parser.completions;
}

std::optional<DynMatcher> Parser::parseMatcherExpression(
    llvm::StringRef &code, const Registry &matcherRegistry,
    const NamedValueMap *namedValues, Diagnostics *error) {
  VariantValue value;
  if (!parseExpression(code, matcherRegistry, namedValues, &value, error))
    return std::nullopt;
  if (!value.isMatcher()) {
    error->addError(SourceRange(), ErrorType::ParserNotAMatcher);
    return std::nullopt;
  }
  std::optional<DynMatcher> result = value.getMatcher().getDynMatcher();
  if (!result) {
    error->addError(SourceRange(), ErrorType::ParserOverloadedType)
        << value.getTypeAsString();
  }
  return result;
}

} // namespace mlir::query::matcher::internal
