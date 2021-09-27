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

#include "AsmParserImpl.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
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
class CustomDialectAsmParser : public AsmParserImpl<DialectAsmParser> {
public:
  CustomDialectAsmParser(StringRef fullSpec, Parser &parser)
      : AsmParserImpl<DialectAsmParser>(parser.getToken().getLoc(), parser),
        fullSpec(fullSpec) {}
  ~CustomDialectAsmParser() override {}

  /// Returns the full specification of the symbol being parsed. This allows
  /// for using a separate parser if necessary.
  StringRef getFullSymbolSpec() const override { return fullSpec; }

private:
  /// The full symbol specification.
  StringRef fullSpec;
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
  ParserState state(sourceMgr, context, symbolState, /*asmState=*/nullptr);
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
        if (Dialect *dialect =
                builder.getContext()->getOrLoadDialect(dialectName)) {
          return parseSymbol<Attribute>(
              symbolData, state.context, state.symbols, [&](Parser &parser) {
                CustomDialectAsmParser customParser(symbolData, parser);
                return dialect->parseAttribute(customParser, attrType);
              });
        }

        // Otherwise, form a new opaque attribute.
        return OpaqueAttr::getChecked(
            [&] { return emitError(loc); },
            Identifier::get(dialectName, state.context), symbolData,
            attrType ? attrType : NoneType::get(state.context));
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
        auto *dialect = state.context->getOrLoadDialect(dialectName);

        if (dialect) {
          return parseSymbol<Type>(
              symbolData, state.context, state.symbols, [&](Parser &parser) {
                CustomDialectAsmParser customParser(symbolData, parser);
                return dialect->parseType(customParser);
              });
        }

        // Otherwise, form a new opaque type.
        return OpaqueType::getChecked(
            [&] { return emitError(loc); },
            Identifier::get(dialectName, state.context), symbolData);
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
