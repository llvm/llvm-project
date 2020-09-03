//===- DialectSymbolParser.cpp - MLIR Dialect Symbol Parser  --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the dialect symbols, such as extended
// attributes and types.
//
//===----------------------------------------------------------------------===//

#include "Parser.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::detail;
using llvm::MemoryBuffer;
using llvm::SMLoc;
using llvm::SourceMgr;

namespace {
/// This class provides the main implementation of the DialectAsmParser that
/// allows for dialects to parse attributes and types. This allows for dialect
/// hooking into the main MLIR parsing logic.
class CustomDialectAsmParser : public DialectAsmParser {
public:
  CustomDialectAsmParser(StringRef fullSpec, Parser &parser)
      : fullSpec(fullSpec), nameLoc(parser.getToken().getLoc()),
        parser(parser) {}
  ~CustomDialectAsmParser() override {}

  /// Emit a diagnostic at the specified location and return failure.
  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) override {
    return parser.emitError(loc, message);
  }

  /// Return a builder which provides useful access to MLIRContext, global
  /// objects like types and attributes.
  Builder &getBuilder() const override { return parser.builder; }

  /// Get the location of the next token and store it into the argument.  This
  /// always succeeds.
  llvm::SMLoc getCurrentLocation() override {
    return parser.getToken().getLoc();
  }

  /// Return the location of the original name token.
  llvm::SMLoc getNameLoc() const override { return nameLoc; }

  /// Re-encode the given source location as an MLIR location and return it.
  Location getEncodedSourceLoc(llvm::SMLoc loc) override {
    return parser.getEncodedSourceLocation(loc);
  }

  /// Returns the full specification of the symbol being parsed. This allows
  /// for using a separate parser if necessary.
  StringRef getFullSymbolSpec() const override { return fullSpec; }

  /// Parse a floating point value from the stream.
  ParseResult parseFloat(double &result) override {
    bool negative = parser.consumeIf(Token::minus);
    Token curTok = parser.getToken();

    // Check for a floating point value.
    if (curTok.is(Token::floatliteral)) {
      auto val = curTok.getFloatingPointValue();
      if (!val.hasValue())
        return emitError(curTok.getLoc(), "floating point value too large");
      parser.consumeToken(Token::floatliteral);
      result = negative ? -*val : *val;
      return success();
    }

    // TODO: support hex floating point values.
    return emitError(getCurrentLocation(), "expected floating point literal");
  }

  /// Parse an optional integer value from the stream.
  OptionalParseResult parseOptionalInteger(uint64_t &result) override {
    Token curToken = parser.getToken();
    if (curToken.isNot(Token::integer, Token::minus))
      return llvm::None;

    bool negative = parser.consumeIf(Token::minus);
    Token curTok = parser.getToken();
    if (parser.parseToken(Token::integer, "expected integer value"))
      return failure();

    auto val = curTok.getUInt64IntegerValue();
    if (!val)
      return emitError(curTok.getLoc(), "integer value too large");
    result = negative ? -*val : *val;
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a `->` token.
  ParseResult parseArrow() override {
    return parser.parseToken(Token::arrow, "expected '->'");
  }

  /// Parses a `->` if present.
  ParseResult parseOptionalArrow() override {
    return success(parser.consumeIf(Token::arrow));
  }

  /// Parse a '{' token.
  ParseResult parseLBrace() override {
    return parser.parseToken(Token::l_brace, "expected '{'");
  }

  /// Parse a '{' token if present
  ParseResult parseOptionalLBrace() override {
    return success(parser.consumeIf(Token::l_brace));
  }

  /// Parse a `}` token.
  ParseResult parseRBrace() override {
    return parser.parseToken(Token::r_brace, "expected '}'");
  }

  /// Parse a `}` token if present
  ParseResult parseOptionalRBrace() override {
    return success(parser.consumeIf(Token::r_brace));
  }

  /// Parse a `:` token.
  ParseResult parseColon() override {
    return parser.parseToken(Token::colon, "expected ':'");
  }

  /// Parse a `:` token if present.
  ParseResult parseOptionalColon() override {
    return success(parser.consumeIf(Token::colon));
  }

  /// Parse a `,` token.
  ParseResult parseComma() override {
    return parser.parseToken(Token::comma, "expected ','");
  }

  /// Parse a `,` token if present.
  ParseResult parseOptionalComma() override {
    return success(parser.consumeIf(Token::comma));
  }

  /// Parses a `...` if present.
  ParseResult parseOptionalEllipsis() override {
    return success(parser.consumeIf(Token::ellipsis));
  }

  /// Parse a `=` token.
  ParseResult parseEqual() override {
    return parser.parseToken(Token::equal, "expected '='");
  }

  /// Parse a `=` token if present.
  ParseResult parseOptionalEqual() override {
    return success(parser.consumeIf(Token::equal));
  }

  /// Parse a '<' token.
  ParseResult parseLess() override {
    return parser.parseToken(Token::less, "expected '<'");
  }

  /// Parse a `<` token if present.
  ParseResult parseOptionalLess() override {
    return success(parser.consumeIf(Token::less));
  }

  /// Parse a '>' token.
  ParseResult parseGreater() override {
    return parser.parseToken(Token::greater, "expected '>'");
  }

  /// Parse a `>` token if present.
  ParseResult parseOptionalGreater() override {
    return success(parser.consumeIf(Token::greater));
  }

  /// Parse a `(` token.
  ParseResult parseLParen() override {
    return parser.parseToken(Token::l_paren, "expected '('");
  }

  /// Parses a '(' if present.
  ParseResult parseOptionalLParen() override {
    return success(parser.consumeIf(Token::l_paren));
  }

  /// Parse a `)` token.
  ParseResult parseRParen() override {
    return parser.parseToken(Token::r_paren, "expected ')'");
  }

  /// Parses a ')' if present.
  ParseResult parseOptionalRParen() override {
    return success(parser.consumeIf(Token::r_paren));
  }

  /// Parse a `[` token.
  ParseResult parseLSquare() override {
    return parser.parseToken(Token::l_square, "expected '['");
  }

  /// Parses a '[' if present.
  ParseResult parseOptionalLSquare() override {
    return success(parser.consumeIf(Token::l_square));
  }

  /// Parse a `]` token.
  ParseResult parseRSquare() override {
    return parser.parseToken(Token::r_square, "expected ']'");
  }

  /// Parses a ']' if present.
  ParseResult parseOptionalRSquare() override {
    return success(parser.consumeIf(Token::r_square));
  }

  /// Parses a '?' if present.
  ParseResult parseOptionalQuestion() override {
    return success(parser.consumeIf(Token::question));
  }

  /// Parses a '*' if present.
  ParseResult parseOptionalStar() override {
    return success(parser.consumeIf(Token::star));
  }

  /// Returns if the current token corresponds to a keyword.
  bool isCurrentTokenAKeyword() const {
    return parser.getToken().is(Token::bare_identifier) ||
           parser.getToken().isKeyword();
  }

  /// Parse the given keyword if present.
  ParseResult parseOptionalKeyword(StringRef keyword) override {
    // Check that the current token has the same spelling.
    if (!isCurrentTokenAKeyword() || parser.getTokenSpelling() != keyword)
      return failure();
    parser.consumeToken();
    return success();
  }

  /// Parse a keyword, if present, into 'keyword'.
  ParseResult parseOptionalKeyword(StringRef *keyword) override {
    // Check that the current token is a keyword.
    if (!isCurrentTokenAKeyword())
      return failure();

    *keyword = parser.getTokenSpelling();
    parser.consumeToken();
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Attribute Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an arbitrary attribute and return it in result.
  ParseResult parseAttribute(Attribute &result, Type type) override {
    result = parser.parseAttribute(type);
    return success(static_cast<bool>(result));
  }

  /// Parse an affine map instance into 'map'.
  ParseResult parseAffineMap(AffineMap &map) override {
    return parser.parseAffineMapReference(map);
  }

  /// Parse an integer set instance into 'set'.
  ParseResult printIntegerSet(IntegerSet &set) override {
    return parser.parseIntegerSetReference(set);
  }

  //===--------------------------------------------------------------------===//
  // Type Parsing
  //===--------------------------------------------------------------------===//

  ParseResult parseType(Type &result) override {
    result = parser.parseType();
    return success(static_cast<bool>(result));
  }

  ParseResult parseDimensionList(SmallVectorImpl<int64_t> &dimensions,
                                 bool allowDynamic) override {
    return parser.parseDimensionListRanked(dimensions, allowDynamic);
  }

private:
  /// The full symbol specification.
  StringRef fullSpec;

  /// The source location of the dialect symbol.
  SMLoc nameLoc;

  /// The main parser.
  Parser &parser;
};
} // namespace

/// Parse the body of a pretty dialect symbol, which starts and ends with <>'s,
/// and may be recursive.  Return with the 'prettyName' StringRef encompassing
/// the entire pretty name.
///
///   pretty-dialect-sym-body ::= '<' pretty-dialect-sym-contents+ '>'
///   pretty-dialect-sym-contents ::= pretty-dialect-sym-body
///                                  | '(' pretty-dialect-sym-contents+ ')'
///                                  | '[' pretty-dialect-sym-contents+ ']'
///                                  | '{' pretty-dialect-sym-contents+ '}'
///                                  | '[^[<({>\])}\0]+'
///
ParseResult Parser::parsePrettyDialectSymbolName(StringRef &prettyName) {
  // Pretty symbol names are a relatively unstructured format that contains a
  // series of properly nested punctuation, with anything else in the middle.
  // Scan ahead to find it and consume it if successful, otherwise emit an
  // error.
  auto *curPtr = getTokenSpelling().data();

  SmallVector<char, 8> nestedPunctuation;

  // Scan over the nested punctuation, bailing out on error and consuming until
  // we find the end.  We know that we're currently looking at the '<', so we
  // can go until we find the matching '>' character.
  assert(*curPtr == '<');
  do {
    char c = *curPtr++;
    switch (c) {
    case '\0':
      // This also handles the EOF case.
      return emitError("unexpected nul or EOF in pretty dialect name");
    case '<':
    case '[':
    case '(':
    case '{':
      nestedPunctuation.push_back(c);
      continue;

    case '-':
      // The sequence `->` is treated as special token.
      if (*curPtr == '>')
        ++curPtr;
      continue;

    case '>':
      if (nestedPunctuation.pop_back_val() != '<')
        return emitError("unbalanced '>' character in pretty dialect name");
      break;
    case ']':
      if (nestedPunctuation.pop_back_val() != '[')
        return emitError("unbalanced ']' character in pretty dialect name");
      break;
    case ')':
      if (nestedPunctuation.pop_back_val() != '(')
        return emitError("unbalanced ')' character in pretty dialect name");
      break;
    case '}':
      if (nestedPunctuation.pop_back_val() != '{')
        return emitError("unbalanced '}' character in pretty dialect name");
      break;

    default:
      continue;
    }
  } while (!nestedPunctuation.empty());

  // Ok, we succeeded, remember where we stopped, reset the lexer to know it is
  // consuming all this stuff, and return.
  state.lex.resetPointer(curPtr);

  unsigned length = curPtr - prettyName.begin();
  prettyName = StringRef(prettyName.begin(), length);
  consumeToken();
  return success();
}

/// Parse an extended dialect symbol.
template <typename Symbol, typename SymbolAliasMap, typename CreateFn>
static Symbol parseExtendedSymbol(Parser &p, Token::Kind identifierTok,
                                  SymbolAliasMap &aliases,
                                  CreateFn &&createSymbol) {
  // Parse the dialect namespace.
  StringRef identifier = p.getTokenSpelling().drop_front();
  auto loc = p.getToken().getLoc();
  p.consumeToken(identifierTok);

  // If there is no '<' token following this, and if the typename contains no
  // dot, then we are parsing a symbol alias.
  if (p.getToken().isNot(Token::less) && !identifier.contains('.')) {
    // Check for an alias for this type.
    auto aliasIt = aliases.find(identifier);
    if (aliasIt == aliases.end())
      return (p.emitError("undefined symbol alias id '" + identifier + "'"),
              nullptr);
    return aliasIt->second;
  }

  // Otherwise, we are parsing a dialect-specific symbol.  If the name contains
  // a dot, then this is the "pretty" form.  If not, it is the verbose form that
  // looks like <"...">.
  std::string symbolData;
  auto dialectName = identifier;

  // Handle the verbose form, where "identifier" is a simple dialect name.
  if (!identifier.contains('.')) {
    // Consume the '<'.
    if (p.parseToken(Token::less, "expected '<' in dialect type"))
      return nullptr;

    // Parse the symbol specific data.
    if (p.getToken().isNot(Token::string))
      return (p.emitError("expected string literal data in dialect symbol"),
              nullptr);
    symbolData = p.getToken().getStringValue();
    loc = llvm::SMLoc::getFromPointer(p.getToken().getLoc().getPointer() + 1);
    p.consumeToken(Token::string);

    // Consume the '>'.
    if (p.parseToken(Token::greater, "expected '>' in dialect symbol"))
      return nullptr;
  } else {
    // Ok, the dialect name is the part of the identifier before the dot, the
    // part after the dot is the dialect's symbol, or the start thereof.
    auto dotHalves = identifier.split('.');
    dialectName = dotHalves.first;
    auto prettyName = dotHalves.second;
    loc = llvm::SMLoc::getFromPointer(prettyName.data());

    // If the dialect's symbol is followed immediately by a <, then lex the body
    // of it into prettyName.
    if (p.getToken().is(Token::less) &&
        prettyName.bytes_end() == p.getTokenSpelling().bytes_begin()) {
      if (p.parsePrettyDialectSymbolName(prettyName))
        return nullptr;
    }

    symbolData = prettyName.str();
  }

  // Record the name location of the type remapped to the top level buffer.
  llvm::SMLoc locInTopLevelBuffer = p.remapLocationToTopLevelBuffer(loc);
  p.getState().symbols.nestedParserLocs.push_back(locInTopLevelBuffer);

  // Call into the provided symbol construction function.
  Symbol sym = createSymbol(dialectName, symbolData, loc);

  // Pop the last parser location.
  p.getState().symbols.nestedParserLocs.pop_back();
  return sym;
}

/// Parses a symbol, of type 'T', and returns it if parsing was successful. If
/// parsing failed, nullptr is returned. The number of bytes read from the input
/// string is returned in 'numRead'.
template <typename T, typename ParserFn>
static T parseSymbol(StringRef inputStr, MLIRContext *context,
                     SymbolState &symbolState, ParserFn &&parserFn,
                     size_t *numRead = nullptr) {
  SourceMgr sourceMgr;
  auto memBuffer = MemoryBuffer::getMemBuffer(
      inputStr, /*BufferName=*/"<mlir_parser_buffer>",
      /*RequiresNullTerminator=*/false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  ParserState state(sourceMgr, context, symbolState);
  Parser parser(state);

  Token startTok = parser.getToken();
  T symbol = parserFn(parser);
  if (!symbol)
    return T();

  // If 'numRead' is valid, then provide the number of bytes that were read.
  Token endTok = parser.getToken();
  if (numRead) {
    *numRead = static_cast<size_t>(endTok.getLoc().getPointer() -
                                   startTok.getLoc().getPointer());

    // Otherwise, ensure that all of the tokens were parsed.
  } else if (startTok.getLoc() != endTok.getLoc() && endTok.isNot(Token::eof)) {
    parser.emitError(endTok.getLoc(), "encountered unexpected token");
    return T();
  }
  return symbol;
}

/// Parse an extended attribute.
///
///   extended-attribute ::= (dialect-attribute | attribute-alias)
///   dialect-attribute  ::= `#` dialect-namespace `<` `"` attr-data `"` `>`
///   dialect-attribute  ::= `#` alias-name pretty-dialect-sym-body?
///   attribute-alias    ::= `#` alias-name
///
Attribute Parser::parseExtendedAttr(Type type) {
  Attribute attr = parseExtendedSymbol<Attribute>(
      *this, Token::hash_identifier, state.symbols.attributeAliasDefinitions,
      [&](StringRef dialectName, StringRef symbolData,
          llvm::SMLoc loc) -> Attribute {
        // Parse an optional trailing colon type.
        Type attrType = type;
        if (consumeIf(Token::colon) && !(attrType = parseType()))
          return Attribute();

        // If we found a registered dialect, then ask it to parse the attribute.
        if (auto *dialect = state.context->getRegisteredDialect(dialectName)) {
          return parseSymbol<Attribute>(
              symbolData, state.context, state.symbols, [&](Parser &parser) {
                CustomDialectAsmParser customParser(symbolData, parser);
                return dialect->parseAttribute(customParser, attrType);
              });
        }

        // Otherwise, form a new opaque attribute.
        return OpaqueAttr::getChecked(
            Identifier::get(dialectName, state.context), symbolData,
            attrType ? attrType : NoneType::get(state.context),
            getEncodedSourceLocation(loc));
      });

  // Ensure that the attribute has the same type as requested.
  if (attr && type && attr.getType() != type) {
    emitError("attribute type different than expected: expected ")
        << type << ", but got " << attr.getType();
    return nullptr;
  }
  return attr;
}

/// Parse an extended type.
///
///   extended-type ::= (dialect-type | type-alias)
///   dialect-type  ::= `!` dialect-namespace `<` `"` type-data `"` `>`
///   dialect-type  ::= `!` alias-name pretty-dialect-attribute-body?
///   type-alias    ::= `!` alias-name
///
Type Parser::parseExtendedType() {
  return parseExtendedSymbol<Type>(
      *this, Token::exclamation_identifier, state.symbols.typeAliasDefinitions,
      [&](StringRef dialectName, StringRef symbolData,
          llvm::SMLoc loc) -> Type {
        // If we found a registered dialect, then ask it to parse the type.
        if (auto *dialect = state.context->getRegisteredDialect(dialectName)) {
          return parseSymbol<Type>(
              symbolData, state.context, state.symbols, [&](Parser &parser) {
                CustomDialectAsmParser customParser(symbolData, parser);
                return dialect->parseType(customParser);
              });
        }

        // Otherwise, form a new opaque type.
        return OpaqueType::getChecked(
            Identifier::get(dialectName, state.context), symbolData,
            state.context, getEncodedSourceLocation(loc));
      });
}

//===----------------------------------------------------------------------===//
// mlir::parseAttribute/parseType
//===----------------------------------------------------------------------===//

/// Parses a symbol, of type 'T', and returns it if parsing was successful. If
/// parsing failed, nullptr is returned. The number of bytes read from the input
/// string is returned in 'numRead'.
template <typename T, typename ParserFn>
static T parseSymbol(StringRef inputStr, MLIRContext *context, size_t &numRead,
                     ParserFn &&parserFn) {
  SymbolState aliasState;
  return parseSymbol<T>(
      inputStr, context, aliasState,
      [&](Parser &parser) {
        SourceMgrDiagnosticHandler handler(
            const_cast<llvm::SourceMgr &>(parser.getSourceMgr()),
            parser.getContext());
        return parserFn(parser);
      },
      &numRead);
}

Attribute mlir::parseAttribute(StringRef attrStr, MLIRContext *context) {
  size_t numRead = 0;
  return parseAttribute(attrStr, context, numRead);
}
Attribute mlir::parseAttribute(StringRef attrStr, Type type) {
  size_t numRead = 0;
  return parseAttribute(attrStr, type, numRead);
}

Attribute mlir::parseAttribute(StringRef attrStr, MLIRContext *context,
                               size_t &numRead) {
  return parseSymbol<Attribute>(attrStr, context, numRead, [](Parser &parser) {
    return parser.parseAttribute();
  });
}
Attribute mlir::parseAttribute(StringRef attrStr, Type type, size_t &numRead) {
  return parseSymbol<Attribute>(
      attrStr, type.getContext(), numRead,
      [type](Parser &parser) { return parser.parseAttribute(type); });
}

Type mlir::parseType(StringRef typeStr, MLIRContext *context) {
  size_t numRead = 0;
  return parseType(typeStr, context, numRead);
}

Type mlir::parseType(StringRef typeStr, MLIRContext *context, size_t &numRead) {
  return parseSymbol<Type>(typeStr, context, numRead,
                           [](Parser &parser) { return parser.parseType(); });
}
