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
  ~CustomDialectAsmParser() override = default;

  /// Returns the full specification of the symbol being parsed. This allows
  /// for using a separate parser if necessary.
  StringRef getFullSymbolSpec() const override { return fullSpec; }

private:
  /// The full symbol specification.
  StringRef fullSpec;
};
} // namespace

///
///   pretty-dialect-sym-body ::= '<' pretty-dialect-sym-contents+ '>'
///   pretty-dialect-sym-contents ::= pretty-dialect-sym-body
///                                  | '(' pretty-dialect-sym-contents+ ')'
///                                  | '[' pretty-dialect-sym-contents+ ']'
///                                  | '{' pretty-dialect-sym-contents+ '}'
///                                  | '[^[<({>\])}\0]+'
///
ParseResult Parser::parseDialectSymbolBody(StringRef &body,
                                           bool &isCodeCompletion) {
  // Symbol bodies are a relatively unstructured format that contains a series
  // of properly nested punctuation, with anything else in the middle. Scan
  // ahead to find it and consume it if successful, otherwise emit an error.
  const char *curPtr = getTokenSpelling().data();

  // Scan over the nested punctuation, bailing out on error and consuming until
  // we find the end. We know that we're currently looking at the '<', so we can
  // go until we find the matching '>' character.
  assert(*curPtr == '<');
  SmallVector<char, 8> nestedPunctuation;
  const char *codeCompleteLoc = state.lex.getCodeCompleteLoc();
  do {
    // Handle code completions, which may appear in the middle of the symbol
    // body.
    if (curPtr == codeCompleteLoc) {
      isCodeCompletion = true;
      nestedPunctuation.clear();
      break;
    }

    char c = *curPtr++;
    switch (c) {
    case '\0':
      // This also handles the EOF case.
      if (!nestedPunctuation.empty()) {
        return emitError() << "unbalanced '" << nestedPunctuation.back()
                           << "' character in pretty dialect name";
      }
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
    case '"': {
      // Dispatch to the lexer to lex past strings.
      resetToken(curPtr - 1);
      curPtr = state.curToken.getEndLoc().getPointer();

      // Handle code completions, which may appear in the middle of the symbol
      // body.
      if (state.curToken.isCodeCompletion()) {
        isCodeCompletion = true;
        nestedPunctuation.clear();
        break;
      }

      // Otherwise, ensure this token was actually a string.
      if (state.curToken.isNot(Token::string))
        return failure();
      break;
    }

    default:
      continue;
    }
  } while (!nestedPunctuation.empty());

  // Ok, we succeeded, remember where we stopped, reset the lexer to know it is
  // consuming all this stuff, and return.
  resetToken(curPtr);

  unsigned length = curPtr - body.begin();
  body = StringRef(body.data(), length);
  return success();
}

/// Parse an extended dialect symbol.
template <typename Symbol, typename SymbolAliasMap, typename CreateFn>
static Symbol parseExtendedSymbol(Parser &p, SymbolAliasMap &aliases,
                                  CreateFn &&createSymbol) {
  Token tok = p.getToken();

  // Handle code completion of the extended symbol.
  StringRef identifier = tok.getSpelling().drop_front();
  if (tok.isCodeCompletion() && identifier.empty())
    return p.codeCompleteDialectSymbol(aliases);

  // Parse the dialect namespace.
  SMLoc loc = p.getToken().getLoc();
  p.consumeToken();

  // Check to see if this is a pretty name.
  auto [dialectName, symbolData] = identifier.split('.');
  bool isPrettyName = !symbolData.empty() || identifier.back() == '.';

  // Check to see if the symbol has trailing data, i.e. has an immediately
  // following '<'.
  bool hasTrailingData =
      p.getToken().is(Token::less) &&
      identifier.bytes_end() == p.getTokenSpelling().bytes_begin();

  // If there is no '<' token following this, and if the typename contains no
  // dot, then we are parsing a symbol alias.
  if (!hasTrailingData && !isPrettyName) {
    // Check for an alias for this type.
    auto aliasIt = aliases.find(identifier);
    if (aliasIt == aliases.end())
      return (p.emitWrongTokenError("undefined symbol alias id '" + identifier +
                                    "'"),
              nullptr);
    return aliasIt->second;
  }

  // If this isn't an alias, we are parsing a dialect-specific symbol. If the
  // name contains a dot, then this is the "pretty" form. If not, it is the
  // verbose form that looks like <...>.
  if (!isPrettyName) {
    // Point the symbol data to the end of the dialect name to start.
    symbolData = StringRef(dialectName.end(), 0);

    // Parse the body of the symbol.
    bool isCodeCompletion = false;
    if (p.parseDialectSymbolBody(symbolData, isCodeCompletion))
      return nullptr;
    symbolData = symbolData.drop_front();

    // If the body contained a code completion it won't have the trailing `>`
    // token, so don't drop it.
    if (!isCodeCompletion)
      symbolData = symbolData.drop_back();
  } else {
    loc = SMLoc::getFromPointer(symbolData.data());

    // If the dialect's symbol is followed immediately by a <, then lex the body
    // of it into prettyName.
    if (hasTrailingData && p.parseDialectSymbolBody(symbolData))
      return nullptr;
  }

  return createSymbol(dialectName, symbolData, loc);
}

/// Parse an extended attribute.
///
///   extended-attribute ::= (dialect-attribute | attribute-alias)
///   dialect-attribute  ::= `#` dialect-namespace `<` attr-data `>`
///                          (`:` type)?
///                        | `#` alias-name pretty-dialect-sym-body? (`:` type)?
///   attribute-alias    ::= `#` alias-name
///
Attribute Parser::parseExtendedAttr(Type type) {
  MLIRContext *ctx = getContext();
  Attribute attr = parseExtendedSymbol<Attribute>(
      *this, state.symbols.attributeAliasDefinitions,
      [&](StringRef dialectName, StringRef symbolData, SMLoc loc) -> Attribute {
        // Parse an optional trailing colon type.
        Type attrType = type;
        if (consumeIf(Token::colon) && !(attrType = parseType()))
          return Attribute();

        // If we found a registered dialect, then ask it to parse the attribute.
        if (Dialect *dialect =
                builder.getContext()->getOrLoadDialect(dialectName)) {
          // Temporarily reset the lexer to let the dialect parse the attribute.
          const char *curLexerPos = getToken().getLoc().getPointer();
          resetToken(symbolData.data());

          // Parse the attribute.
          CustomDialectAsmParser customParser(symbolData, *this);
          Attribute attr = dialect->parseAttribute(customParser, attrType);
          resetToken(curLexerPos);
          return attr;
        }

        // Otherwise, form a new opaque attribute.
        return OpaqueAttr::getChecked(
            [&] { return emitError(loc); }, StringAttr::get(ctx, dialectName),
            symbolData, attrType ? attrType : NoneType::get(ctx));
      });

  // Ensure that the attribute has the same type as requested.
  auto typedAttr = attr.dyn_cast_or_null<TypedAttr>();
  if (type && typedAttr && typedAttr.getType() != type) {
    emitError("attribute type different than expected: expected ")
        << type << ", but got " << typedAttr.getType();
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
  MLIRContext *ctx = getContext();
  return parseExtendedSymbol<Type>(
      *this, state.symbols.typeAliasDefinitions,
      [&](StringRef dialectName, StringRef symbolData, SMLoc loc) -> Type {
        // If we found a registered dialect, then ask it to parse the type.
        if (auto *dialect = ctx->getOrLoadDialect(dialectName)) {
          // Temporarily reset the lexer to let the dialect parse the type.
          const char *curLexerPos = getToken().getLoc().getPointer();
          resetToken(symbolData.data());

          // Parse the type.
          CustomDialectAsmParser customParser(symbolData, *this);
          Type type = dialect->parseType(customParser);
          resetToken(curLexerPos);
          return type;
        }

        // Otherwise, form a new opaque type.
        return OpaqueType::getChecked([&] { return emitError(loc); },
                                      StringAttr::get(ctx, dialectName),
                                      symbolData);
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
  SourceMgr sourceMgr;
  auto memBuffer = MemoryBuffer::getMemBuffer(
      inputStr, /*BufferName=*/"<mlir_parser_buffer>",
      /*RequiresNullTerminator=*/false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  SymbolState aliasState;
  ParserConfig config(context);
  ParserState state(sourceMgr, config, aliasState, /*asmState=*/nullptr,
                    /*codeCompleteContext=*/nullptr);
  Parser parser(state);

  SourceMgrDiagnosticHandler handler(
      const_cast<llvm::SourceMgr &>(parser.getSourceMgr()),
      parser.getContext());
  Token startTok = parser.getToken();
  T symbol = parserFn(parser);
  if (!symbol)
    return T();

  // Provide the number of bytes that were read.
  Token endTok = parser.getToken();
  numRead = static_cast<size_t>(endTok.getLoc().getPointer() -
                                startTok.getLoc().getPointer());
  return symbol;
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
