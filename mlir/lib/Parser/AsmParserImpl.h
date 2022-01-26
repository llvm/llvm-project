//===- AsmParserImpl.h - MLIR AsmParserImpl Class ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_PARSER_ASMPARSERIMPL_H
#define MLIR_LIB_PARSER_ASMPARSERIMPL_H

#include "Parser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/AsmParserState.h"

namespace mlir {
namespace detail {
//===----------------------------------------------------------------------===//
// AsmParserImpl
//===----------------------------------------------------------------------===//

/// This class provides the implementation of the generic parser methods within
/// AsmParser.
template <typename BaseT>
class AsmParserImpl : public BaseT {
public:
  AsmParserImpl(SMLoc nameLoc, Parser &parser)
      : nameLoc(nameLoc), parser(parser) {}
  ~AsmParserImpl() override = default;

  /// Return the location of the original name token.
  SMLoc getNameLoc() const override { return nameLoc; }

  //===--------------------------------------------------------------------===//
  // Utilities
  //===--------------------------------------------------------------------===//

  /// Return if any errors were emitted during parsing.
  bool didEmitError() const { return emittedError; }

  /// Emit a diagnostic at the specified location and return failure.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message) override {
    emittedError = true;
    return parser.emitError(loc, message);
  }

  /// Return a builder which provides useful access to MLIRContext, global
  /// objects like types and attributes.
  Builder &getBuilder() const override { return parser.builder; }

  /// Get the location of the next token and store it into the argument.  This
  /// always succeeds.
  SMLoc getCurrentLocation() override {
    return parser.getToken().getLoc();
  }

  /// Re-encode the given source location as an MLIR location and return it.
  Location getEncodedSourceLoc(SMLoc loc) override {
    return parser.getEncodedSourceLocation(loc);
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  using Delimiter = AsmParser::Delimiter;

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

  /// Parses a '?' token.
  ParseResult parseQuestion() override {
    return parser.parseToken(Token::question, "expected '?'");
  }

  /// Parses a '?' if present.
  ParseResult parseOptionalQuestion() override {
    return success(parser.consumeIf(Token::question));
  }

  /// Parses a '*' token.
  ParseResult parseStar() override {
    return parser.parseToken(Token::star, "expected '*'");
  }

  /// Parses a '*' if present.
  ParseResult parseOptionalStar() override {
    return success(parser.consumeIf(Token::star));
  }

  /// Parses a '+' token.
  ParseResult parsePlus() override {
    return parser.parseToken(Token::plus, "expected '+'");
  }

  /// Parses a '+' token if present.
  ParseResult parseOptionalPlus() override {
    return success(parser.consumeIf(Token::plus));
  }

  /// Parses a quoted string token if present.
  ParseResult parseOptionalString(std::string *string) override {
    if (!parser.getToken().is(Token::string))
      return failure();

    if (string)
      *string = parser.getToken().getStringValue();
    parser.consumeToken();
    return success();
  }

  /// Returns true if the current token corresponds to a keyword.
  bool isCurrentTokenAKeyword() const {
    return parser.getToken().isAny(Token::bare_identifier, Token::inttype) ||
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

  /// Parse a keyword if it is one of the 'allowedKeywords'.
  ParseResult
  parseOptionalKeyword(StringRef *keyword,
                       ArrayRef<StringRef> allowedKeywords) override {
    // Check that the current token is a keyword.
    if (!isCurrentTokenAKeyword())
      return failure();

    StringRef currentKeyword = parser.getTokenSpelling();
    if (llvm::is_contained(allowedKeywords, currentKeyword)) {
      *keyword = currentKeyword;
      parser.consumeToken();
      return success();
    }

    return failure();
  }

  /// Parse an optional keyword or string and set instance into 'result'.`
  ParseResult parseOptionalKeywordOrString(std::string *result) override {
    StringRef keyword;
    if (succeeded(parseOptionalKeyword(&keyword))) {
      *result = keyword.str();
      return success();
    }

    return parseOptionalString(result);
  }

  /// Parse a floating point value from the stream.
  ParseResult parseFloat(double &result) override {
    bool isNegative = parser.consumeIf(Token::minus);
    Token curTok = parser.getToken();
    SMLoc loc = curTok.getLoc();

    // Check for a floating point value.
    if (curTok.is(Token::floatliteral)) {
      auto val = curTok.getFloatingPointValue();
      if (!val.hasValue())
        return emitError(loc, "floating point value too large");
      parser.consumeToken(Token::floatliteral);
      result = isNegative ? -*val : *val;
      return success();
    }

    // Check for a hexadecimal float value.
    if (curTok.is(Token::integer)) {
      Optional<APFloat> apResult;
      if (failed(parser.parseFloatFromIntegerLiteral(
              apResult, curTok, isNegative, APFloat::IEEEdouble(),
              /*typeSizeInBits=*/64)))
        return failure();

      parser.consumeToken(Token::integer);
      result = apResult->convertToDouble();
      return success();
    }

    return emitError(loc, "expected floating point literal");
  }

  /// Parse an optional integer value from the stream.
  OptionalParseResult parseOptionalInteger(APInt &result) override {
    return parser.parseOptionalInteger(result);
  }

  /// Parse a list of comma-separated items with an optional delimiter.  If a
  /// delimiter is provided, then an empty list is allowed.  If not, then at
  /// least one element will be parsed.
  ParseResult parseCommaSeparatedList(Delimiter delimiter,
                                      function_ref<ParseResult()> parseElt,
                                      StringRef contextMessage) override {
    return parser.parseCommaSeparatedList(delimiter, parseElt, contextMessage);
  }

  //===--------------------------------------------------------------------===//
  // Attribute Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an arbitrary attribute and return it in result.
  ParseResult parseAttribute(Attribute &result, Type type) override {
    result = parser.parseAttribute(type);
    return success(static_cast<bool>(result));
  }

  /// Parse a custom attribute with the provided callback, unless the next
  /// token is `#`, in which case the generic parser is invoked.
  ParseResult parseCustomAttributeWithFallback(
      Attribute &result, Type type,
      function_ref<ParseResult(Attribute &result, Type type)> parseAttribute)
      override {
    if (parser.getToken().isNot(Token::hash_identifier))
      return parseAttribute(result, type);
    result = parser.parseAttribute(type);
    return success(static_cast<bool>(result));
  }

  /// Parse a custom attribute with the provided callback, unless the next
  /// token is `#`, in which case the generic parser is invoked.
  ParseResult parseCustomTypeWithFallback(
      Type &result,
      function_ref<ParseResult(Type &result)> parseType) override {
    if (parser.getToken().isNot(Token::exclamation_identifier))
      return parseType(result);
    result = parser.parseType();
    return success(static_cast<bool>(result));
  }

  OptionalParseResult parseOptionalAttribute(Attribute &result,
                                             Type type) override {
    return parser.parseOptionalAttribute(result, type);
  }
  OptionalParseResult parseOptionalAttribute(ArrayAttr &result,
                                             Type type) override {
    return parser.parseOptionalAttribute(result, type);
  }
  OptionalParseResult parseOptionalAttribute(StringAttr &result,
                                             Type type) override {
    return parser.parseOptionalAttribute(result, type);
  }

  /// Parse a named dictionary into 'result' if it is present.
  ParseResult parseOptionalAttrDict(NamedAttrList &result) override {
    if (parser.getToken().isNot(Token::l_brace))
      return success();
    return parser.parseAttributeDict(result);
  }

  /// Parse a named dictionary into 'result' if the `attributes` keyword is
  /// present.
  ParseResult parseOptionalAttrDictWithKeyword(NamedAttrList &result) override {
    if (failed(parseOptionalKeyword("attributes")))
      return success();
    return parser.parseAttributeDict(result);
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
  // Identifier Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an optional @-identifier and store it (without the '@' symbol) in a
  /// string attribute named 'attrName'.
  ParseResult parseOptionalSymbolName(StringAttr &result, StringRef attrName,
                                      NamedAttrList &attrs) override {
    Token atToken = parser.getToken();
    if (atToken.isNot(Token::at_identifier))
      return failure();

    result = getBuilder().getStringAttr(atToken.getSymbolReference());
    attrs.push_back(getBuilder().getNamedAttr(attrName, result));
    parser.consumeToken();

    // If we are populating the assembly parser state, record this as a symbol
    // reference.
    if (parser.getState().asmState) {
      parser.getState().asmState->addUses(SymbolRefAttr::get(result),
                                          atToken.getLocRange());
    }
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Type Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a type.
  ParseResult parseType(Type &result) override {
    return failure(!(result = parser.parseType()));
  }

  /// Parse an optional type.
  OptionalParseResult parseOptionalType(Type &result) override {
    return parser.parseOptionalType(result);
  }

  /// Parse an arrow followed by a type list.
  ParseResult parseArrowTypeList(SmallVectorImpl<Type> &result) override {
    if (parseArrow() || parser.parseFunctionResultTypes(result))
      return failure();
    return success();
  }

  /// Parse an optional arrow followed by a type list.
  ParseResult
  parseOptionalArrowTypeList(SmallVectorImpl<Type> &result) override {
    if (!parser.consumeIf(Token::arrow))
      return success();
    return parser.parseFunctionResultTypes(result);
  }

  /// Parse a colon followed by a type.
  ParseResult parseColonType(Type &result) override {
    return failure(parser.parseToken(Token::colon, "expected ':'") ||
                   !(result = parser.parseType()));
  }

  /// Parse a colon followed by a type list, which must have at least one type.
  ParseResult parseColonTypeList(SmallVectorImpl<Type> &result) override {
    if (parser.parseToken(Token::colon, "expected ':'"))
      return failure();
    return parser.parseTypeListNoParens(result);
  }

  /// Parse an optional colon followed by a type list, which if present must
  /// have at least one type.
  ParseResult
  parseOptionalColonTypeList(SmallVectorImpl<Type> &result) override {
    if (!parser.consumeIf(Token::colon))
      return success();
    return parser.parseTypeListNoParens(result);
  }

  ParseResult parseDimensionList(SmallVectorImpl<int64_t> &dimensions,
                                 bool allowDynamic) override {
    return parser.parseDimensionListRanked(dimensions, allowDynamic);
  }

  ParseResult parseXInDimensionList() override {
    return parser.parseXInDimensionList();
  }

protected:
  /// The source location of the dialect symbol.
  SMLoc nameLoc;

  /// The main parser.
  Parser &parser;

  /// A flag that indicates if any errors were emitted during parsing.
  bool emittedError = false;
};
} // namespace detail
} // namespace mlir

#endif // MLIR_LIB_PARSER_ASMPARSERIMPL_H
