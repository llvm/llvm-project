//===- Parser.cpp - MLIR Parser Implementation ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the MLIR textual form.
//
//===----------------------------------------------------------------------===//

#include "Parser.h"
#include "AsmParserImpl.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/AsmParser/CodeComplete.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// CodeComplete
//===----------------------------------------------------------------------===//

AsmParserCodeCompleteContext::~AsmParserCodeCompleteContext() = default;

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// Parse a list of comma-separated items with an optional delimiter.  If a
/// delimiter is provided, then an empty list is allowed.  If not, then at
/// least one element will be parsed.
ParseResult
Parser::parseCommaSeparatedList(Delimiter delimiter,
                                function_ref<ParseResult()> parseElementFn,
                                StringRef contextMessage) {
  switch (delimiter) {
  case Delimiter::None:
    break;
  case Delimiter::OptionalParen:
    if (getToken().isNot(Token::l_paren))
      return success();
    LLVM_FALLTHROUGH;
  case Delimiter::Paren:
    if (parseToken(Token::l_paren, "expected '('" + contextMessage))
      return failure();
    // Check for empty list.
    if (consumeIf(Token::r_paren))
      return success();
    break;
  case Delimiter::OptionalLessGreater:
    // Check for absent list.
    if (getToken().isNot(Token::less))
      return success();
    LLVM_FALLTHROUGH;
  case Delimiter::LessGreater:
    if (parseToken(Token::less, "expected '<'" + contextMessage))
      return success();
    // Check for empty list.
    if (consumeIf(Token::greater))
      return success();
    break;
  case Delimiter::OptionalSquare:
    if (getToken().isNot(Token::l_square))
      return success();
    LLVM_FALLTHROUGH;
  case Delimiter::Square:
    if (parseToken(Token::l_square, "expected '['" + contextMessage))
      return failure();
    // Check for empty list.
    if (consumeIf(Token::r_square))
      return success();
    break;
  case Delimiter::OptionalBraces:
    if (getToken().isNot(Token::l_brace))
      return success();
    LLVM_FALLTHROUGH;
  case Delimiter::Braces:
    if (parseToken(Token::l_brace, "expected '{'" + contextMessage))
      return failure();
    // Check for empty list.
    if (consumeIf(Token::r_brace))
      return success();
    break;
  }

  // Non-empty case starts with an element.
  if (parseElementFn())
    return failure();

  // Otherwise we have a list of comma separated elements.
  while (consumeIf(Token::comma)) {
    if (parseElementFn())
      return failure();
  }

  switch (delimiter) {
  case Delimiter::None:
    return success();
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

/// Parse a comma-separated list of elements, terminated with an arbitrary
/// token.  This allows empty lists if allowEmptyList is true.
///
///   abstract-list ::= rightToken                  // if allowEmptyList == true
///   abstract-list ::= element (',' element)* rightToken
///
ParseResult
Parser::parseCommaSeparatedListUntil(Token::Kind rightToken,
                                     function_ref<ParseResult()> parseElement,
                                     bool allowEmptyList) {
  // Handle the empty case.
  if (getToken().is(rightToken)) {
    if (!allowEmptyList)
      return emitWrongTokenError("expected list element");
    consumeToken(rightToken);
    return success();
  }

  if (parseCommaSeparatedList(parseElement) ||
      parseToken(rightToken, "expected ',' or '" +
                                 Token::getTokenSpelling(rightToken) + "'"))
    return failure();

  return success();
}

InFlightDiagnostic Parser::emitError(const Twine &message) {
  auto loc = state.curToken.getLoc();
  if (state.curToken.isNot(Token::eof))
    return emitError(loc, message);

  // If the error is to be emitted at EOF, move it back one character.
  return emitError(SMLoc::getFromPointer(loc.getPointer() - 1), message);
}

InFlightDiagnostic Parser::emitError(SMLoc loc, const Twine &message) {
  auto diag = mlir::emitError(getEncodedSourceLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(Token::error))
    diag.abandon();
  return diag;
}

/// Emit an error about a "wrong token".  If the current token is at the
/// start of a source line, this will apply heuristics to back up and report
/// the error at the end of the previous line, which is where the expected
/// token is supposed to be.
InFlightDiagnostic Parser::emitWrongTokenError(const Twine &message) {
  auto loc = state.curToken.getLoc();

  // If the error is to be emitted at EOF, move it back one character.
  if (state.curToken.is(Token::eof))
    loc = SMLoc::getFromPointer(loc.getPointer() - 1);

  // This is the location we were originally asked to report the error at.
  auto originalLoc = loc;

  // Determine if the token is at the start of the current line.
  const char *bufferStart = state.lex.getBufferBegin();
  const char *curPtr = loc.getPointer();

  // Use this StringRef to keep track of what we are going to back up through,
  // it provides nicer string search functions etc.
  StringRef startOfBuffer(bufferStart, curPtr - bufferStart);

  // Back up over entirely blank lines.
  while (true) {
    // Back up until we see a \n, but don't look past the buffer start.
    startOfBuffer = startOfBuffer.rtrim(" \t");

    // For tokens with no preceding source line, just emit at the original
    // location.
    if (startOfBuffer.empty())
      return emitError(originalLoc, message);

    // If we found something that isn't the end of line, then we're done.
    if (startOfBuffer.back() != '\n' && startOfBuffer.back() != '\r')
      return emitError(SMLoc::getFromPointer(startOfBuffer.end()), message);

    // Drop the \n so we emit the diagnostic at the end of the line.
    startOfBuffer = startOfBuffer.drop_back();

    // Check to see if the preceding line has a comment on it.  We assume that a
    // `//` is the start of a comment, which is mostly correct.
    // TODO: This will do the wrong thing for // in a string literal.
    auto prevLine = startOfBuffer;
    size_t newLineIndex = prevLine.find_last_of("\n\r");
    if (newLineIndex != StringRef::npos)
      prevLine = prevLine.drop_front(newLineIndex);

    // If we find a // in the current line, then emit the diagnostic before it.
    size_t commentStart = prevLine.find("//");
    if (commentStart != StringRef::npos)
      startOfBuffer = startOfBuffer.drop_back(prevLine.size() - commentStart);
  }
}

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult Parser::parseToken(Token::Kind expectedToken,
                               const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitWrongTokenError(message);
}

/// Parse an optional integer value from the stream.
OptionalParseResult Parser::parseOptionalInteger(APInt &result) {
  // Parse `false` and `true` keywords as 0 and 1 respectively.
  if (consumeIf(Token::kw_false)) {
    result = false;
    return success();
  } else if (consumeIf(Token::kw_true)) {
    result = true;
    return success();
  }

  Token curToken = getToken();
  if (curToken.isNot(Token::integer, Token::minus))
    return llvm::None;

  bool negative = consumeIf(Token::minus);
  Token curTok = getToken();
  if (parseToken(Token::integer, "expected integer value"))
    return failure();

  StringRef spelling = curTok.getSpelling();
  bool isHex = spelling.size() > 1 && spelling[1] == 'x';
  if (spelling.getAsInteger(isHex ? 0 : 10, result))
    return emitError(curTok.getLoc(), "integer value too large");

  // Make sure we have a zero at the top so we return the right signedness.
  if (result.isNegative())
    result = result.zext(result.getBitWidth() + 1);

  // Process the negative sign if present.
  if (negative)
    result.negate();

  return success();
}

/// Parse a floating point value from an integer literal token.
ParseResult Parser::parseFloatFromIntegerLiteral(
    Optional<APFloat> &result, const Token &tok, bool isNegative,
    const llvm::fltSemantics &semantics, size_t typeSizeInBits) {
  SMLoc loc = tok.getLoc();
  StringRef spelling = tok.getSpelling();
  bool isHex = spelling.size() > 1 && spelling[1] == 'x';
  if (!isHex) {
    return emitError(loc, "unexpected decimal integer literal for a "
                          "floating point value")
               .attachNote()
           << "add a trailing dot to make the literal a float";
  }
  if (isNegative) {
    return emitError(loc, "hexadecimal float literal should not have a "
                          "leading minus");
  }

  Optional<uint64_t> value = tok.getUInt64IntegerValue();
  if (!value)
    return emitError(loc, "hexadecimal float constant out of range for type");

  if (&semantics == &APFloat::IEEEdouble()) {
    result = APFloat(semantics, APInt(typeSizeInBits, *value));
    return success();
  }

  APInt apInt(typeSizeInBits, *value);
  if (apInt != *value)
    return emitError(loc, "hexadecimal float constant out of range for type");
  result = APFloat(semantics, apInt);

  return success();
}

ParseResult Parser::parseOptionalKeyword(StringRef *keyword) {
  // Check that the current token is a keyword.
  if (!isCurrentTokenAKeyword())
    return failure();

  *keyword = getTokenSpelling();
  consumeToken();
  return success();
}

//===----------------------------------------------------------------------===//
// Resource Parsing

FailureOr<AsmDialectResourceHandle>
Parser::parseResourceHandle(const OpAsmDialectInterface *dialect,
                            StringRef &name) {
  assert(dialect && "expected valid dialect interface");
  SMLoc nameLoc = getToken().getLoc();
  if (failed(parseOptionalKeyword(&name)))
    return emitError("expected identifier key for 'resource' entry");
  auto &resources = getState().symbols.dialectResources;

  // If this is the first time encountering this handle, ask the dialect to
  // resolve a reference to this handle. This allows for us to remap the name of
  // the handle if necessary.
  std::pair<std::string, AsmDialectResourceHandle> &entry =
      resources[dialect][name];
  if (entry.first.empty()) {
    FailureOr<AsmDialectResourceHandle> result = dialect->declareResource(name);
    if (failed(result)) {
      return emitError(nameLoc)
             << "unknown 'resource' key '" << name << "' for dialect '"
             << dialect->getDialect()->getNamespace() << "'";
    }
    entry.first = dialect->getResourceKey(*result);
    entry.second = *result;
  }

  name = entry.first;
  return entry.second;
}

FailureOr<AsmDialectResourceHandle>
Parser::parseResourceHandle(Dialect *dialect) {
  const auto *interface = dyn_cast<OpAsmDialectInterface>(dialect);
  if (!interface) {
    return emitError() << "dialect '" << dialect->getNamespace()
                       << "' does not expect resource handles";
  }
  StringRef resourceName;
  return parseResourceHandle(interface, resourceName);
}

//===----------------------------------------------------------------------===//
// Code Completion

ParseResult Parser::codeCompleteDialectName() {
  state.codeCompleteContext->completeDialectName();
  return failure();
}

ParseResult Parser::codeCompleteOperationName(StringRef dialectName) {
  // Perform some simple validation on the dialect name. This doesn't need to be
  // extensive, it's more of an optimization (to avoid checking completion
  // results when we know they will fail).
  if (dialectName.empty() || dialectName.contains('.'))
    return failure();
  state.codeCompleteContext->completeOperationName(dialectName);
  return failure();
}

ParseResult Parser::codeCompleteDialectOrElidedOpName(SMLoc loc) {
  // Check to see if there is anything else on the current line. This check
  // isn't strictly necessary, but it does avoid unnecessarily triggering
  // completions for operations and dialects in situations where we don't want
  // them (e.g. at the end of an operation).
  auto shouldIgnoreOpCompletion = [&]() {
    const char *bufBegin = state.lex.getBufferBegin();
    const char *it = loc.getPointer() - 1;
    for (; it > bufBegin && *it != '\n'; --it)
      if (!llvm::is_contained(StringRef(" \t\r"), *it))
        return true;
    return false;
  };
  if (shouldIgnoreOpCompletion())
    return failure();

  // The completion here is either for a dialect name, or an operation name
  // whose dialect prefix was elided. For this we simply invoke both of the
  // individual completion methods.
  (void)codeCompleteDialectName();
  return codeCompleteOperationName(state.defaultDialectStack.back());
}

ParseResult Parser::codeCompleteStringDialectOrOperationName(StringRef name) {
  // If the name is empty, this is the start of the string and contains the
  // dialect.
  if (name.empty())
    return codeCompleteDialectName();

  // Otherwise, we treat this as completing an operation name. The current name
  // is used as the dialect namespace.
  if (name.consume_back("."))
    return codeCompleteOperationName(name);
  return failure();
}

ParseResult Parser::codeCompleteExpectedTokens(ArrayRef<StringRef> tokens) {
  state.codeCompleteContext->completeExpectedTokens(tokens, /*optional=*/false);
  return failure();
}
ParseResult Parser::codeCompleteOptionalTokens(ArrayRef<StringRef> tokens) {
  state.codeCompleteContext->completeExpectedTokens(tokens, /*optional=*/true);
  return failure();
}

Attribute Parser::codeCompleteAttribute() {
  state.codeCompleteContext->completeAttribute(
      state.symbols.attributeAliasDefinitions);
  return {};
}
Type Parser::codeCompleteType() {
  state.codeCompleteContext->completeType(state.symbols.typeAliasDefinitions);
  return {};
}

Attribute
Parser::codeCompleteDialectSymbol(const llvm::StringMap<Attribute> &aliases) {
  state.codeCompleteContext->completeDialectAttributeOrAlias(aliases);
  return {};
}
Type Parser::codeCompleteDialectSymbol(const llvm::StringMap<Type> &aliases) {
  state.codeCompleteContext->completeDialectTypeOrAlias(aliases);
  return {};
}

//===----------------------------------------------------------------------===//
// OperationParser
//===----------------------------------------------------------------------===//

namespace {
/// This class provides support for parsing operations and regions of
/// operations.
class OperationParser : public Parser {
public:
  OperationParser(ParserState &state, ModuleOp topLevelOp);
  ~OperationParser();

  /// After parsing is finished, this function must be called to see if there
  /// are any remaining issues.
  ParseResult finalize();

  //===--------------------------------------------------------------------===//
  // SSA Value Handling
  //===--------------------------------------------------------------------===//

  using UnresolvedOperand = OpAsmParser::UnresolvedOperand;
  using Argument = OpAsmParser::Argument;

  struct DeferredLocInfo {
    SMLoc loc;
    StringRef identifier;
  };

  /// Push a new SSA name scope to the parser.
  void pushSSANameScope(bool isIsolated);

  /// Pop the last SSA name scope from the parser.
  ParseResult popSSANameScope();

  /// Register a definition of a value with the symbol table.
  ParseResult addDefinition(UnresolvedOperand useInfo, Value value);

  /// Parse an optional list of SSA uses into 'results'.
  ParseResult
  parseOptionalSSAUseList(SmallVectorImpl<UnresolvedOperand> &results);

  /// Parse a single SSA use into 'result'.  If 'allowResultNumber' is true then
  /// we allow #42 syntax.
  ParseResult parseSSAUse(UnresolvedOperand &result,
                          bool allowResultNumber = true);

  /// Given a reference to an SSA value and its type, return a reference. This
  /// returns null on failure.
  Value resolveSSAUse(UnresolvedOperand useInfo, Type type);

  ParseResult parseSSADefOrUseAndType(
      function_ref<ParseResult(UnresolvedOperand, Type)> action);

  ParseResult parseOptionalSSAUseAndTypeList(SmallVectorImpl<Value> &results);

  /// Return the location of the value identified by its name and number if it
  /// has been already reference.
  Optional<SMLoc> getReferenceLoc(StringRef name, unsigned number) {
    auto &values = isolatedNameScopes.back().values;
    if (!values.count(name) || number >= values[name].size())
      return {};
    if (values[name][number].value)
      return values[name][number].loc;
    return {};
  }

  //===--------------------------------------------------------------------===//
  // Operation Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an operation instance.
  ParseResult parseOperation();

  /// Parse a single operation successor.
  ParseResult parseSuccessor(Block *&dest);

  /// Parse a comma-separated list of operation successors in brackets.
  ParseResult parseSuccessors(SmallVectorImpl<Block *> &destinations);

  /// Parse an operation instance that is in the generic form.
  Operation *parseGenericOperation();

  /// Parse different components, viz., use-info of operand(s), successor(s),
  /// region(s), attribute(s) and function-type, of the generic form of an
  /// operation instance and populate the input operation-state 'result' with
  /// those components. If any of the components is explicitly provided, then
  /// skip parsing that component.
  ParseResult parseGenericOperationAfterOpName(
      OperationState &result,
      Optional<ArrayRef<UnresolvedOperand>> parsedOperandUseInfo = llvm::None,
      Optional<ArrayRef<Block *>> parsedSuccessors = llvm::None,
      Optional<MutableArrayRef<std::unique_ptr<Region>>> parsedRegions =
          llvm::None,
      Optional<ArrayRef<NamedAttribute>> parsedAttributes = llvm::None,
      Optional<FunctionType> parsedFnType = llvm::None);

  /// Parse an operation instance that is in the generic form and insert it at
  /// the provided insertion point.
  Operation *parseGenericOperation(Block *insertBlock,
                                   Block::iterator insertPt);

  /// This type is used to keep track of things that are either an Operation or
  /// a BlockArgument.  We cannot use Value for this, because not all Operations
  /// have results.
  using OpOrArgument = llvm::PointerUnion<Operation *, BlockArgument>;

  /// Parse an optional trailing location and add it to the specifier Operation
  /// or `UnresolvedOperand` if present.
  ///
  ///   trailing-location ::= (`loc` (`(` location `)` | attribute-alias))?
  ///
  ParseResult parseTrailingLocationSpecifier(OpOrArgument opOrArgument);

  /// Parse a location alias, that is a sequence looking like: #loc42
  /// The alias may have already be defined or may be defined later, in which
  /// case an OpaqueLoc is used a placeholder.
  ParseResult parseLocationAlias(LocationAttr &loc);

  /// This is the structure of a result specifier in the assembly syntax,
  /// including the name, number of results, and location.
  using ResultRecord = std::tuple<StringRef, unsigned, SMLoc>;

  /// Parse an operation instance that is in the op-defined custom form.
  /// resultInfo specifies information about the "%name =" specifiers.
  Operation *parseCustomOperation(ArrayRef<ResultRecord> resultIDs);

  /// Parse the name of an operation, in the custom form. On success, return a
  /// an object of type 'OperationName'. Otherwise, failure is returned.
  FailureOr<OperationName> parseCustomOperationName();

  //===--------------------------------------------------------------------===//
  // Region Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a region into 'region' with the provided entry block arguments.
  /// 'isIsolatedNameScope' indicates if the naming scope of this region is
  /// isolated from those above.
  ParseResult parseRegion(Region &region, ArrayRef<Argument> entryArguments,
                          bool isIsolatedNameScope = false);

  /// Parse a region body into 'region'.
  ParseResult parseRegionBody(Region &region, SMLoc startLoc,
                              ArrayRef<Argument> entryArguments,
                              bool isIsolatedNameScope);

  //===--------------------------------------------------------------------===//
  // Block Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a new block into 'block'.
  ParseResult parseBlock(Block *&block);

  /// Parse a list of operations into 'block'.
  ParseResult parseBlockBody(Block *block);

  /// Parse a (possibly empty) list of block arguments.
  ParseResult parseOptionalBlockArgList(Block *owner);

  /// Get the block with the specified name, creating it if it doesn't
  /// already exist.  The location specified is the point of use, which allows
  /// us to diagnose references to blocks that are not defined precisely.
  Block *getBlockNamed(StringRef name, SMLoc loc);

  //===--------------------------------------------------------------------===//
  // Code Completion
  //===--------------------------------------------------------------------===//

  /// The set of various code completion methods. Every completion method
  /// returns `failure` to stop the parsing process after providing completion
  /// results.

  ParseResult codeCompleteSSAUse();
  ParseResult codeCompleteBlock();

private:
  /// This class represents a definition of a Block.
  struct BlockDefinition {
    /// A pointer to the defined Block.
    Block *block;
    /// The location that the Block was defined at.
    SMLoc loc;
  };
  /// This class represents a definition of a Value.
  struct ValueDefinition {
    /// A pointer to the defined Value.
    Value value;
    /// The location that the Value was defined at.
    SMLoc loc;
  };

  /// Returns the info for a block at the current scope for the given name.
  BlockDefinition &getBlockInfoByName(StringRef name) {
    return blocksByName.back()[name];
  }

  /// Insert a new forward reference to the given block.
  void insertForwardRef(Block *block, SMLoc loc) {
    forwardRef.back().try_emplace(block, loc);
  }

  /// Erase any forward reference to the given block.
  bool eraseForwardRef(Block *block) { return forwardRef.back().erase(block); }

  /// Record that a definition was added at the current scope.
  void recordDefinition(StringRef def);

  /// Get the value entry for the given SSA name.
  SmallVectorImpl<ValueDefinition> &getSSAValueEntry(StringRef name);

  /// Create a forward reference placeholder value with the given location and
  /// result type.
  Value createForwardRefPlaceholder(SMLoc loc, Type type);

  /// Return true if this is a forward reference.
  bool isForwardRefPlaceholder(Value value) {
    return forwardRefPlaceholders.count(value);
  }

  /// This struct represents an isolated SSA name scope. This scope may contain
  /// other nested non-isolated scopes. These scopes are used for operations
  /// that are known to be isolated to allow for reusing names within their
  /// regions, even if those names are used above.
  struct IsolatedSSANameScope {
    /// Record that a definition was added at the current scope.
    void recordDefinition(StringRef def) {
      definitionsPerScope.back().insert(def);
    }

    /// Push a nested name scope.
    void pushSSANameScope() { definitionsPerScope.push_back({}); }

    /// Pop a nested name scope.
    void popSSANameScope() {
      for (auto &def : definitionsPerScope.pop_back_val())
        values.erase(def.getKey());
    }

    /// This keeps track of all of the SSA values we are tracking for each name
    /// scope, indexed by their name. This has one entry per result number.
    llvm::StringMap<SmallVector<ValueDefinition, 1>> values;

    /// This keeps track of all of the values defined by a specific name scope.
    SmallVector<llvm::StringSet<>, 2> definitionsPerScope;
  };

  /// A list of isolated name scopes.
  SmallVector<IsolatedSSANameScope, 2> isolatedNameScopes;

  /// This keeps track of the block names as well as the location of the first
  /// reference for each nested name scope. This is used to diagnose invalid
  /// block references and memorize them.
  SmallVector<DenseMap<StringRef, BlockDefinition>, 2> blocksByName;
  SmallVector<DenseMap<Block *, SMLoc>, 2> forwardRef;

  /// These are all of the placeholders we've made along with the location of
  /// their first reference, to allow checking for use of undefined values.
  DenseMap<Value, SMLoc> forwardRefPlaceholders;

  /// Deffered locations: when parsing `loc(#loc42)` we add an entry to this
  /// map. After parsing the definition `#loc42 = ...` we'll patch back users
  /// of this location.
  std::vector<DeferredLocInfo> deferredLocsReferences;

  /// The builder used when creating parsed operation instances.
  OpBuilder opBuilder;

  /// The top level operation that holds all of the parsed operations.
  Operation *topLevelOp;
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(OperationParser::DeferredLocInfo *)
MLIR_DEFINE_EXPLICIT_TYPE_ID(OperationParser::DeferredLocInfo *)

OperationParser::OperationParser(ParserState &state, ModuleOp topLevelOp)
    : Parser(state), opBuilder(topLevelOp.getRegion()), topLevelOp(topLevelOp) {
  // The top level operation starts a new name scope.
  pushSSANameScope(/*isIsolated=*/true);

  // If we are populating the parser state, prepare it for parsing.
  if (state.asmState)
    state.asmState->initialize(topLevelOp);
}

OperationParser::~OperationParser() {
  for (auto &fwd : forwardRefPlaceholders) {
    // Drop all uses of undefined forward declared reference and destroy
    // defining operation.
    fwd.first.dropAllUses();
    fwd.first.getDefiningOp()->destroy();
  }
  for (const auto &scope : forwardRef) {
    for (const auto &fwd : scope) {
      // Delete all blocks that were created as forward references but never
      // included into a region.
      fwd.first->dropAllUses();
      delete fwd.first;
    }
  }
}

/// After parsing is finished, this function must be called to see if there are
/// any remaining issues.
ParseResult OperationParser::finalize() {
  // Check for any forward references that are left.  If we find any, error
  // out.
  if (!forwardRefPlaceholders.empty()) {
    SmallVector<const char *, 4> errors;
    // Iteration over the map isn't deterministic, so sort by source location.
    for (auto entry : forwardRefPlaceholders)
      errors.push_back(entry.second.getPointer());
    llvm::array_pod_sort(errors.begin(), errors.end());

    for (const char *entry : errors) {
      auto loc = SMLoc::getFromPointer(entry);
      emitError(loc, "use of undeclared SSA value name");
    }
    return failure();
  }

  // Resolve the locations of any deferred operations.
  auto &attributeAliases = state.symbols.attributeAliasDefinitions;
  auto locID = TypeID::get<DeferredLocInfo *>();
  auto resolveLocation = [&, this](auto &opOrArgument) -> LogicalResult {
    auto fwdLoc = opOrArgument.getLoc().template dyn_cast<OpaqueLoc>();
    if (!fwdLoc || fwdLoc.getUnderlyingTypeID() != locID)
      return success();
    auto locInfo = deferredLocsReferences[fwdLoc.getUnderlyingLocation()];
    Attribute attr = attributeAliases.lookup(locInfo.identifier);
    if (!attr)
      return this->emitError(locInfo.loc)
             << "operation location alias was never defined";
    auto locAttr = attr.dyn_cast<LocationAttr>();
    if (!locAttr)
      return this->emitError(locInfo.loc)
             << "expected location, but found '" << attr << "'";
    opOrArgument.setLoc(locAttr);
    return success();
  };

  auto walkRes = topLevelOp->walk([&](Operation *op) {
    if (failed(resolveLocation(*op)))
      return WalkResult::interrupt();
    for (Region &region : op->getRegions())
      for (Block &block : region.getBlocks())
        for (BlockArgument arg : block.getArguments())
          if (failed(resolveLocation(arg)))
            return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkRes.wasInterrupted())
    return failure();

  // Pop the top level name scope.
  if (failed(popSSANameScope()))
    return failure();

  // Verify that the parsed operations are valid.
  if (failed(verify(topLevelOp)))
    return failure();

  // If we are populating the parser state, finalize the top-level operation.
  if (state.asmState)
    state.asmState->finalize(topLevelOp);
  return success();
}

//===----------------------------------------------------------------------===//
// SSA Value Handling
//===----------------------------------------------------------------------===//

void OperationParser::pushSSANameScope(bool isIsolated) {
  blocksByName.push_back(DenseMap<StringRef, BlockDefinition>());
  forwardRef.push_back(DenseMap<Block *, SMLoc>());

  // Push back a new name definition scope.
  if (isIsolated)
    isolatedNameScopes.push_back({});
  isolatedNameScopes.back().pushSSANameScope();
}

ParseResult OperationParser::popSSANameScope() {
  auto forwardRefInCurrentScope = forwardRef.pop_back_val();

  // Verify that all referenced blocks were defined.
  if (!forwardRefInCurrentScope.empty()) {
    SmallVector<std::pair<const char *, Block *>, 4> errors;
    // Iteration over the map isn't deterministic, so sort by source location.
    for (auto entry : forwardRefInCurrentScope) {
      errors.push_back({entry.second.getPointer(), entry.first});
      // Add this block to the top-level region to allow for automatic cleanup.
      topLevelOp->getRegion(0).push_back(entry.first);
    }
    llvm::array_pod_sort(errors.begin(), errors.end());

    for (auto entry : errors) {
      auto loc = SMLoc::getFromPointer(entry.first);
      emitError(loc, "reference to an undefined block");
    }
    return failure();
  }

  // Pop the next nested namescope. If there is only one internal namescope,
  // just pop the isolated scope.
  auto &currentNameScope = isolatedNameScopes.back();
  if (currentNameScope.definitionsPerScope.size() == 1)
    isolatedNameScopes.pop_back();
  else
    currentNameScope.popSSANameScope();

  blocksByName.pop_back();
  return success();
}

/// Register a definition of a value with the symbol table.
ParseResult OperationParser::addDefinition(UnresolvedOperand useInfo,
                                           Value value) {
  auto &entries = getSSAValueEntry(useInfo.name);

  // Make sure there is a slot for this value.
  if (entries.size() <= useInfo.number)
    entries.resize(useInfo.number + 1);

  // If we already have an entry for this, check to see if it was a definition
  // or a forward reference.
  if (auto existing = entries[useInfo.number].value) {
    if (!isForwardRefPlaceholder(existing)) {
      return emitError(useInfo.location)
          .append("redefinition of SSA value '", useInfo.name, "'")
          .attachNote(getEncodedSourceLocation(entries[useInfo.number].loc))
          .append("previously defined here");
    }

    if (existing.getType() != value.getType()) {
      return emitError(useInfo.location)
          .append("definition of SSA value '", useInfo.name, "#",
                  useInfo.number, "' has type ", value.getType())
          .attachNote(getEncodedSourceLocation(entries[useInfo.number].loc))
          .append("previously used here with type ", existing.getType());
    }

    // If it was a forward reference, update everything that used it to use
    // the actual definition instead, delete the forward ref, and remove it
    // from our set of forward references we track.
    existing.replaceAllUsesWith(value);
    existing.getDefiningOp()->destroy();
    forwardRefPlaceholders.erase(existing);

    // If a definition of the value already exists, replace it in the assembly
    // state.
    if (state.asmState)
      state.asmState->refineDefinition(existing, value);
  }

  /// Record this definition for the current scope.
  entries[useInfo.number] = {value, useInfo.location};
  recordDefinition(useInfo.name);
  return success();
}

/// Parse a (possibly empty) list of SSA operands.
///
///   ssa-use-list ::= ssa-use (`,` ssa-use)*
///   ssa-use-list-opt ::= ssa-use-list?
///
ParseResult OperationParser::parseOptionalSSAUseList(
    SmallVectorImpl<UnresolvedOperand> &results) {
  if (!getToken().isOrIsCodeCompletionFor(Token::percent_identifier))
    return success();
  return parseCommaSeparatedList([&]() -> ParseResult {
    UnresolvedOperand result;
    if (parseSSAUse(result))
      return failure();
    results.push_back(result);
    return success();
  });
}

/// Parse a SSA operand for an operation.
///
///   ssa-use ::= ssa-id
///
ParseResult OperationParser::parseSSAUse(UnresolvedOperand &result,
                                         bool allowResultNumber) {
  if (getToken().isCodeCompletion())
    return codeCompleteSSAUse();

  result.name = getTokenSpelling();
  result.number = 0;
  result.location = getToken().getLoc();
  if (parseToken(Token::percent_identifier, "expected SSA operand"))
    return failure();

  // If we have an attribute ID, it is a result number.
  if (getToken().is(Token::hash_identifier)) {
    if (!allowResultNumber)
      return emitError("result number not allowed in argument list");

    if (auto value = getToken().getHashIdentifierNumber())
      result.number = *value;
    else
      return emitError("invalid SSA value result number");
    consumeToken(Token::hash_identifier);
  }

  return success();
}

/// Given an unbound reference to an SSA value and its type, return the value
/// it specifies.  This returns null on failure.
Value OperationParser::resolveSSAUse(UnresolvedOperand useInfo, Type type) {
  auto &entries = getSSAValueEntry(useInfo.name);

  // Functor used to record the use of the given value if the assembly state
  // field is populated.
  auto maybeRecordUse = [&](Value value) {
    if (state.asmState)
      state.asmState->addUses(value, useInfo.location);
    return value;
  };

  // If we have already seen a value of this name, return it.
  if (useInfo.number < entries.size() && entries[useInfo.number].value) {
    Value result = entries[useInfo.number].value;
    // Check that the type matches the other uses.
    if (result.getType() == type)
      return maybeRecordUse(result);

    emitError(useInfo.location, "use of value '")
        .append(useInfo.name,
                "' expects different type than prior uses: ", type, " vs ",
                result.getType())
        .attachNote(getEncodedSourceLocation(entries[useInfo.number].loc))
        .append("prior use here");
    return nullptr;
  }

  // Make sure we have enough slots for this.
  if (entries.size() <= useInfo.number)
    entries.resize(useInfo.number + 1);

  // If the value has already been defined and this is an overly large result
  // number, diagnose that.
  if (entries[0].value && !isForwardRefPlaceholder(entries[0].value))
    return (emitError(useInfo.location, "reference to invalid result number"),
            nullptr);

  // Otherwise, this is a forward reference.  Create a placeholder and remember
  // that we did so.
  Value result = createForwardRefPlaceholder(useInfo.location, type);
  entries[useInfo.number] = {result, useInfo.location};
  return maybeRecordUse(result);
}

/// Parse an SSA use with an associated type.
///
///   ssa-use-and-type ::= ssa-use `:` type
ParseResult OperationParser::parseSSADefOrUseAndType(
    function_ref<ParseResult(UnresolvedOperand, Type)> action) {
  UnresolvedOperand useInfo;
  if (parseSSAUse(useInfo) ||
      parseToken(Token::colon, "expected ':' and type for SSA operand"))
    return failure();

  auto type = parseType();
  if (!type)
    return failure();

  return action(useInfo, type);
}

/// Parse a (possibly empty) list of SSA operands, followed by a colon, then
/// followed by a type list.
///
///   ssa-use-and-type-list
///     ::= ssa-use-list ':' type-list-no-parens
///
ParseResult OperationParser::parseOptionalSSAUseAndTypeList(
    SmallVectorImpl<Value> &results) {
  SmallVector<UnresolvedOperand, 4> valueIDs;
  if (parseOptionalSSAUseList(valueIDs))
    return failure();

  // If there were no operands, then there is no colon or type lists.
  if (valueIDs.empty())
    return success();

  SmallVector<Type, 4> types;
  if (parseToken(Token::colon, "expected ':' in operand list") ||
      parseTypeListNoParens(types))
    return failure();

  if (valueIDs.size() != types.size())
    return emitError("expected ")
           << valueIDs.size() << " types to match operand list";

  results.reserve(valueIDs.size());
  for (unsigned i = 0, e = valueIDs.size(); i != e; ++i) {
    if (auto value = resolveSSAUse(valueIDs[i], types[i]))
      results.push_back(value);
    else
      return failure();
  }

  return success();
}

/// Record that a definition was added at the current scope.
void OperationParser::recordDefinition(StringRef def) {
  isolatedNameScopes.back().recordDefinition(def);
}

/// Get the value entry for the given SSA name.
auto OperationParser::getSSAValueEntry(StringRef name)
    -> SmallVectorImpl<ValueDefinition> & {
  return isolatedNameScopes.back().values[name];
}

/// Create and remember a new placeholder for a forward reference.
Value OperationParser::createForwardRefPlaceholder(SMLoc loc, Type type) {
  // Forward references are always created as operations, because we just need
  // something with a def/use chain.
  //
  // We create these placeholders as having an empty name, which we know
  // cannot be created through normal user input, allowing us to distinguish
  // them.
  auto name = OperationName("builtin.unrealized_conversion_cast", getContext());
  auto *op = Operation::create(
      getEncodedSourceLocation(loc), name, type, /*operands=*/{},
      /*attributes=*/llvm::None, /*successors=*/{}, /*numRegions=*/0);
  forwardRefPlaceholders[op->getResult(0)] = loc;
  return op->getResult(0);
}

//===----------------------------------------------------------------------===//
// Operation Parsing
//===----------------------------------------------------------------------===//

/// Parse an operation.
///
///  operation         ::= op-result-list?
///                        (generic-operation | custom-operation)
///                        trailing-location?
///  generic-operation ::= string-literal `(` ssa-use-list? `)`
///                        successor-list? (`(` region-list `)`)?
///                        attribute-dict? `:` function-type
///  custom-operation  ::= bare-id custom-operation-format
///  op-result-list    ::= op-result (`,` op-result)* `=`
///  op-result         ::= ssa-id (`:` integer-literal)
///
ParseResult OperationParser::parseOperation() {
  auto loc = getToken().getLoc();
  SmallVector<ResultRecord, 1> resultIDs;
  size_t numExpectedResults = 0;
  if (getToken().is(Token::percent_identifier)) {
    // Parse the group of result ids.
    auto parseNextResult = [&]() -> ParseResult {
      // Parse the next result id.
      Token nameTok = getToken();
      if (parseToken(Token::percent_identifier,
                     "expected valid ssa identifier"))
        return failure();

      // If the next token is a ':', we parse the expected result count.
      size_t expectedSubResults = 1;
      if (consumeIf(Token::colon)) {
        // Check that the next token is an integer.
        if (!getToken().is(Token::integer))
          return emitWrongTokenError("expected integer number of results");

        // Check that number of results is > 0.
        auto val = getToken().getUInt64IntegerValue();
        if (!val || *val < 1)
          return emitError(
              "expected named operation to have at least 1 result");
        consumeToken(Token::integer);
        expectedSubResults = *val;
      }

      resultIDs.emplace_back(nameTok.getSpelling(), expectedSubResults,
                             nameTok.getLoc());
      numExpectedResults += expectedSubResults;
      return success();
    };
    if (parseCommaSeparatedList(parseNextResult))
      return failure();

    if (parseToken(Token::equal, "expected '=' after SSA name"))
      return failure();
  }

  Operation *op;
  Token nameTok = getToken();
  if (nameTok.is(Token::bare_identifier) || nameTok.isKeyword())
    op = parseCustomOperation(resultIDs);
  else if (nameTok.is(Token::string))
    op = parseGenericOperation();
  else if (nameTok.isCodeCompletionFor(Token::string))
    return codeCompleteStringDialectOrOperationName(nameTok.getStringValue());
  else if (nameTok.isCodeCompletion())
    return codeCompleteDialectOrElidedOpName(loc);
  else
    return emitWrongTokenError("expected operation name in quotes");

  // If parsing of the basic operation failed, then this whole thing fails.
  if (!op)
    return failure();

  // If the operation had a name, register it.
  if (!resultIDs.empty()) {
    if (op->getNumResults() == 0)
      return emitError(loc, "cannot name an operation with no results");
    if (numExpectedResults != op->getNumResults())
      return emitError(loc, "operation defines ")
             << op->getNumResults() << " results but was provided "
             << numExpectedResults << " to bind";

    // Add this operation to the assembly state if it was provided to populate.
    if (state.asmState) {
      unsigned resultIt = 0;
      SmallVector<std::pair<unsigned, SMLoc>> asmResultGroups;
      asmResultGroups.reserve(resultIDs.size());
      for (ResultRecord &record : resultIDs) {
        asmResultGroups.emplace_back(resultIt, std::get<2>(record));
        resultIt += std::get<1>(record);
      }
      state.asmState->finalizeOperationDefinition(
          op, nameTok.getLocRange(), /*endLoc=*/getToken().getLoc(),
          asmResultGroups);
    }

    // Add definitions for each of the result groups.
    unsigned opResI = 0;
    for (ResultRecord &resIt : resultIDs) {
      for (unsigned subRes : llvm::seq<unsigned>(0, std::get<1>(resIt))) {
        if (addDefinition({std::get<2>(resIt), std::get<0>(resIt), subRes},
                          op->getResult(opResI++)))
          return failure();
      }
    }

    // Add this operation to the assembly state if it was provided to populate.
  } else if (state.asmState) {
    state.asmState->finalizeOperationDefinition(op, nameTok.getLocRange(),
                                                /*endLoc=*/getToken().getLoc());
  }

  return success();
}

/// Parse a single operation successor.
///
///   successor ::= block-id
///
ParseResult OperationParser::parseSuccessor(Block *&dest) {
  if (getToken().isCodeCompletion())
    return codeCompleteBlock();

  // Verify branch is identifier and get the matching block.
  if (!getToken().is(Token::caret_identifier))
    return emitWrongTokenError("expected block name");
  dest = getBlockNamed(getTokenSpelling(), getToken().getLoc());
  consumeToken();
  return success();
}

/// Parse a comma-separated list of operation successors in brackets.
///
///   successor-list ::= `[` successor (`,` successor )* `]`
///
ParseResult
OperationParser::parseSuccessors(SmallVectorImpl<Block *> &destinations) {
  if (parseToken(Token::l_square, "expected '['"))
    return failure();

  auto parseElt = [this, &destinations] {
    Block *dest;
    ParseResult res = parseSuccessor(dest);
    destinations.push_back(dest);
    return res;
  };
  return parseCommaSeparatedListUntil(Token::r_square, parseElt,
                                      /*allowEmptyList=*/false);
}

namespace {
// RAII-style guard for cleaning up the regions in the operation state before
// deleting them.  Within the parser, regions may get deleted if parsing failed,
// and other errors may be present, in particular undominated uses.  This makes
// sure such uses are deleted.
struct CleanupOpStateRegions {
  ~CleanupOpStateRegions() {
    SmallVector<Region *, 4> regionsToClean;
    regionsToClean.reserve(state.regions.size());
    for (auto &region : state.regions)
      if (region)
        for (auto &block : *region)
          block.dropAllDefinedValueUses();
  }
  OperationState &state;
};
} // namespace

ParseResult OperationParser::parseGenericOperationAfterOpName(
    OperationState &result,
    Optional<ArrayRef<UnresolvedOperand>> parsedOperandUseInfo,
    Optional<ArrayRef<Block *>> parsedSuccessors,
    Optional<MutableArrayRef<std::unique_ptr<Region>>> parsedRegions,
    Optional<ArrayRef<NamedAttribute>> parsedAttributes,
    Optional<FunctionType> parsedFnType) {

  // Parse the operand list, if not explicitly provided.
  SmallVector<UnresolvedOperand, 8> opInfo;
  if (!parsedOperandUseInfo) {
    if (parseToken(Token::l_paren, "expected '(' to start operand list") ||
        parseOptionalSSAUseList(opInfo) ||
        parseToken(Token::r_paren, "expected ')' to end operand list")) {
      return failure();
    }
    parsedOperandUseInfo = opInfo;
  }

  // Parse the successor list, if not explicitly provided.
  if (!parsedSuccessors) {
    if (getToken().is(Token::l_square)) {
      // Check if the operation is not a known terminator.
      if (!result.name.mightHaveTrait<OpTrait::IsTerminator>())
        return emitError("successors in non-terminator");

      SmallVector<Block *, 2> successors;
      if (parseSuccessors(successors))
        return failure();
      result.addSuccessors(successors);
    }
  } else {
    result.addSuccessors(*parsedSuccessors);
  }

  // Parse the region list, if not explicitly provided.
  if (!parsedRegions) {
    if (consumeIf(Token::l_paren)) {
      do {
        // Create temporary regions with the top level region as parent.
        result.regions.emplace_back(new Region(topLevelOp));
        if (parseRegion(*result.regions.back(), /*entryArguments=*/{}))
          return failure();
      } while (consumeIf(Token::comma));
      if (parseToken(Token::r_paren, "expected ')' to end region list"))
        return failure();
    }
  } else {
    result.addRegions(*parsedRegions);
  }

  // Parse the attributes, if not explicitly provided.
  if (!parsedAttributes) {
    if (getToken().is(Token::l_brace)) {
      if (parseAttributeDict(result.attributes))
        return failure();
    }
  } else {
    result.addAttributes(*parsedAttributes);
  }

  // Parse the operation type, if not explicitly provided.
  Location typeLoc = result.location;
  if (!parsedFnType) {
    if (parseToken(Token::colon, "expected ':' followed by operation type"))
      return failure();

    typeLoc = getEncodedSourceLocation(getToken().getLoc());
    auto type = parseType();
    if (!type)
      return failure();
    auto fnType = type.dyn_cast<FunctionType>();
    if (!fnType)
      return mlir::emitError(typeLoc, "expected function type");

    parsedFnType = fnType;
  }

  result.addTypes(parsedFnType->getResults());

  // Check that we have the right number of types for the operands.
  ArrayRef<Type> operandTypes = parsedFnType->getInputs();
  if (operandTypes.size() != parsedOperandUseInfo->size()) {
    auto plural = "s"[parsedOperandUseInfo->size() == 1];
    return mlir::emitError(typeLoc, "expected ")
           << parsedOperandUseInfo->size() << " operand type" << plural
           << " but had " << operandTypes.size();
  }

  // Resolve all of the operands.
  for (unsigned i = 0, e = parsedOperandUseInfo->size(); i != e; ++i) {
    result.operands.push_back(
        resolveSSAUse((*parsedOperandUseInfo)[i], operandTypes[i]));
    if (!result.operands.back())
      return failure();
  }

  return success();
}

Operation *OperationParser::parseGenericOperation() {
  // Get location information for the operation.
  auto srcLocation = getEncodedSourceLocation(getToken().getLoc());

  std::string name = getToken().getStringValue();
  if (name.empty())
    return (emitError("empty operation name is invalid"), nullptr);
  if (name.find('\0') != StringRef::npos)
    return (emitError("null character not allowed in operation name"), nullptr);

  consumeToken(Token::string);

  OperationState result(srcLocation, name);
  CleanupOpStateRegions guard{result};

  // Lazy load dialects in the context as needed.
  if (!result.name.isRegistered()) {
    StringRef dialectName = StringRef(name).split('.').first;
    if (!getContext()->getLoadedDialect(dialectName) &&
        !getContext()->getOrLoadDialect(dialectName) &&
        !getContext()->allowsUnregisteredDialects()) {
      // Emit an error if the dialect couldn't be loaded (i.e., it was not
      // registered) and unregistered dialects aren't allowed.
      emitError("operation being parsed with an unregistered dialect. If "
                "this is intended, please use -allow-unregistered-dialect "
                "with the MLIR tool used");
      return nullptr;
    }
  }

  // If we are populating the parser state, start a new operation definition.
  if (state.asmState)
    state.asmState->startOperationDefinition(result.name);

  if (parseGenericOperationAfterOpName(result))
    return nullptr;

  // Create the operation and try to parse a location for it.
  Operation *op = opBuilder.create(result);
  if (parseTrailingLocationSpecifier(op))
    return nullptr;
  return op;
}

Operation *OperationParser::parseGenericOperation(Block *insertBlock,
                                                  Block::iterator insertPt) {
  Token nameToken = getToken();

  OpBuilder::InsertionGuard restoreInsertionPoint(opBuilder);
  opBuilder.setInsertionPoint(insertBlock, insertPt);
  Operation *op = parseGenericOperation();
  if (!op)
    return nullptr;

  // If we are populating the parser asm state, finalize this operation
  // definition.
  if (state.asmState)
    state.asmState->finalizeOperationDefinition(op, nameToken.getLocRange(),
                                                /*endLoc=*/getToken().getLoc());
  return op;
}

namespace {
class CustomOpAsmParser : public AsmParserImpl<OpAsmParser> {
public:
  CustomOpAsmParser(
      SMLoc nameLoc, ArrayRef<OperationParser::ResultRecord> resultIDs,
      function_ref<ParseResult(OpAsmParser &, OperationState &)> parseAssembly,
      bool isIsolatedFromAbove, StringRef opName, OperationParser &parser)
      : AsmParserImpl<OpAsmParser>(nameLoc, parser), resultIDs(resultIDs),
        parseAssembly(parseAssembly), isIsolatedFromAbove(isIsolatedFromAbove),
        opName(opName), parser(parser) {
    (void)isIsolatedFromAbove; // Only used in assert, silence unused warning.
  }

  /// Parse an instance of the operation described by 'opDefinition' into the
  /// provided operation state.
  ParseResult parseOperation(OperationState &opState) {
    if (parseAssembly(*this, opState))
      return failure();
    // Verify that the parsed attributes does not have duplicate attributes.
    // This can happen if an attribute set during parsing is also specified in
    // the attribute dictionary in the assembly, or the attribute is set
    // multiple during parsing.
    Optional<NamedAttribute> duplicate = opState.attributes.findDuplicate();
    if (duplicate)
      return emitError(getNameLoc(), "attribute '")
             << duplicate->getName().getValue()
             << "' occurs more than once in the attribute list";
    return success();
  }

  Operation *parseGenericOperation(Block *insertBlock,
                                   Block::iterator insertPt) final {
    return parser.parseGenericOperation(insertBlock, insertPt);
  }

  FailureOr<OperationName> parseCustomOperationName() final {
    return parser.parseCustomOperationName();
  }

  ParseResult parseGenericOperationAfterOpName(
      OperationState &result,
      Optional<ArrayRef<UnresolvedOperand>> parsedUnresolvedOperands,
      Optional<ArrayRef<Block *>> parsedSuccessors,
      Optional<MutableArrayRef<std::unique_ptr<Region>>> parsedRegions,
      Optional<ArrayRef<NamedAttribute>> parsedAttributes,
      Optional<FunctionType> parsedFnType) final {
    return parser.parseGenericOperationAfterOpName(
        result, parsedUnresolvedOperands, parsedSuccessors, parsedRegions,
        parsedAttributes, parsedFnType);
  }
  //===--------------------------------------------------------------------===//
  // Utilities
  //===--------------------------------------------------------------------===//

  /// Return the name of the specified result in the specified syntax, as well
  /// as the subelement in the name.  For example, in this operation:
  ///
  ///  %x, %y:2, %z = foo.op
  ///
  ///    getResultName(0) == {"x", 0 }
  ///    getResultName(1) == {"y", 0 }
  ///    getResultName(2) == {"y", 1 }
  ///    getResultName(3) == {"z", 0 }
  std::pair<StringRef, unsigned>
  getResultName(unsigned resultNo) const override {
    // Scan for the resultID that contains this result number.
    for (const auto &entry : resultIDs) {
      if (resultNo < std::get<1>(entry)) {
        // Don't pass on the leading %.
        StringRef name = std::get<0>(entry).drop_front();
        return {name, resultNo};
      }
      resultNo -= std::get<1>(entry);
    }

    // Invalid result number.
    return {"", ~0U};
  }

  /// Return the number of declared SSA results.  This returns 4 for the foo.op
  /// example in the comment for getResultName.
  size_t getNumResults() const override {
    size_t count = 0;
    for (auto &entry : resultIDs)
      count += std::get<1>(entry);
    return count;
  }

  /// Emit a diagnostic at the specified location and return failure.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message) override {
    return AsmParserImpl<OpAsmParser>::emitError(loc, "custom op '" + opName +
                                                          "' " + message);
  }

  //===--------------------------------------------------------------------===//
  // Operand Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a single operand.
  ParseResult parseOperand(UnresolvedOperand &result,
                           bool allowResultNumber = true) override {
    OperationParser::UnresolvedOperand useInfo;
    if (parser.parseSSAUse(useInfo, allowResultNumber))
      return failure();

    result = {useInfo.location, useInfo.name, useInfo.number};
    return success();
  }

  /// Parse a single operand if present.
  OptionalParseResult
  parseOptionalOperand(UnresolvedOperand &result,
                       bool allowResultNumber = true) override {
    if (parser.getToken().isOrIsCodeCompletionFor(Token::percent_identifier))
      return parseOperand(result, allowResultNumber);
    return llvm::None;
  }

  /// Parse zero or more SSA comma-separated operand references with a specified
  /// surrounding delimiter, and an optional required operand count.
  ParseResult parseOperandList(SmallVectorImpl<UnresolvedOperand> &result,
                               Delimiter delimiter = Delimiter::None,
                               bool allowResultNumber = true,
                               int requiredOperandCount = -1) override {
    // The no-delimiter case has some special handling for better diagnostics.
    if (delimiter == Delimiter::None) {
      // parseCommaSeparatedList doesn't handle the missing case for "none",
      // so we handle it custom here.
      Token tok = parser.getToken();
      if (!tok.isOrIsCodeCompletionFor(Token::percent_identifier)) {
        // If we didn't require any operands or required exactly zero (weird)
        // then this is success.
        if (requiredOperandCount == -1 || requiredOperandCount == 0)
          return success();

        // Otherwise, try to produce a nice error message.
        if (tok.isAny(Token::l_paren, Token::l_square))
          return parser.emitError("unexpected delimiter");
        return parser.emitWrongTokenError("expected operand");
      }
    }

    auto parseOneOperand = [&]() -> ParseResult {
      return parseOperand(result.emplace_back(), allowResultNumber);
    };

    auto startLoc = parser.getToken().getLoc();
    if (parseCommaSeparatedList(delimiter, parseOneOperand, " in operand list"))
      return failure();

    // Check that we got the expected # of elements.
    if (requiredOperandCount != -1 &&
        result.size() != static_cast<size_t>(requiredOperandCount))
      return emitError(startLoc, "expected ")
             << requiredOperandCount << " operands";
    return success();
  }

  /// Resolve an operand to an SSA value, emitting an error on failure.
  ParseResult resolveOperand(const UnresolvedOperand &operand, Type type,
                             SmallVectorImpl<Value> &result) override {
    if (auto value = parser.resolveSSAUse(operand, type)) {
      result.push_back(value);
      return success();
    }
    return failure();
  }

  /// Parse an AffineMap of SSA ids.
  ParseResult
  parseAffineMapOfSSAIds(SmallVectorImpl<UnresolvedOperand> &operands,
                         Attribute &mapAttr, StringRef attrName,
                         NamedAttrList &attrs, Delimiter delimiter) override {
    SmallVector<UnresolvedOperand, 2> dimOperands;
    SmallVector<UnresolvedOperand, 1> symOperands;

    auto parseElement = [&](bool isSymbol) -> ParseResult {
      UnresolvedOperand operand;
      if (parseOperand(operand))
        return failure();
      if (isSymbol)
        symOperands.push_back(operand);
      else
        dimOperands.push_back(operand);
      return success();
    };

    AffineMap map;
    if (parser.parseAffineMapOfSSAIds(map, parseElement, delimiter))
      return failure();
    // Add AffineMap attribute.
    if (map) {
      mapAttr = AffineMapAttr::get(map);
      attrs.push_back(parser.builder.getNamedAttr(attrName, mapAttr));
    }

    // Add dim operands before symbol operands in 'operands'.
    operands.assign(dimOperands.begin(), dimOperands.end());
    operands.append(symOperands.begin(), symOperands.end());
    return success();
  }

  /// Parse an AffineExpr of SSA ids.
  ParseResult
  parseAffineExprOfSSAIds(SmallVectorImpl<UnresolvedOperand> &dimOperands,
                          SmallVectorImpl<UnresolvedOperand> &symbOperands,
                          AffineExpr &expr) override {
    auto parseElement = [&](bool isSymbol) -> ParseResult {
      UnresolvedOperand operand;
      if (parseOperand(operand))
        return failure();
      if (isSymbol)
        symbOperands.push_back(operand);
      else
        dimOperands.push_back(operand);
      return success();
    };

    return parser.parseAffineExprOfSSAIds(expr, parseElement);
  }

  //===--------------------------------------------------------------------===//
  // Argument Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a single argument with the following syntax:
  ///
  ///   `%ssaname : !type { optionalAttrDict} loc(optionalSourceLoc)`
  ///
  /// If `allowType` is false or `allowAttrs` are false then the respective
  /// parts of the grammar are not parsed.
  ParseResult parseArgument(Argument &result, bool allowType = false,
                            bool allowAttrs = false) override {
    NamedAttrList attrs;
    if (parseOperand(result.ssaName, /*allowResultNumber=*/false) ||
        (allowType && parseColonType(result.type)) ||
        (allowAttrs && parseOptionalAttrDict(attrs)) ||
        parseOptionalLocationSpecifier(result.sourceLoc))
      return failure();
    result.attrs = attrs.getDictionary(getContext());
    return success();
  }

  /// Parse a single argument if present.
  OptionalParseResult parseOptionalArgument(Argument &result, bool allowType,
                                            bool allowAttrs) override {
    if (parser.getToken().is(Token::percent_identifier))
      return parseArgument(result, allowType, allowAttrs);
    return llvm::None;
  }

  ParseResult parseArgumentList(SmallVectorImpl<Argument> &result,
                                Delimiter delimiter, bool allowType,
                                bool allowAttrs) override {
    // The no-delimiter case has some special handling for the empty case.
    if (delimiter == Delimiter::None &&
        parser.getToken().isNot(Token::percent_identifier))
      return success();

    auto parseOneArgument = [&]() -> ParseResult {
      return parseArgument(result.emplace_back(), allowType, allowAttrs);
    };
    return parseCommaSeparatedList(delimiter, parseOneArgument,
                                   " in argument list");
  }

  //===--------------------------------------------------------------------===//
  // Region Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a region that takes `arguments` of `argTypes` types.  This
  /// effectively defines the SSA values of `arguments` and assigns their type.
  ParseResult parseRegion(Region &region, ArrayRef<Argument> arguments,
                          bool enableNameShadowing) override {
    // Try to parse the region.
    (void)isIsolatedFromAbove;
    assert((!enableNameShadowing || isIsolatedFromAbove) &&
           "name shadowing is only allowed on isolated regions");
    if (parser.parseRegion(region, arguments, enableNameShadowing))
      return failure();
    return success();
  }

  /// Parses a region if present.
  OptionalParseResult parseOptionalRegion(Region &region,
                                          ArrayRef<Argument> arguments,
                                          bool enableNameShadowing) override {
    if (parser.getToken().isNot(Token::l_brace))
      return llvm::None;
    return parseRegion(region, arguments, enableNameShadowing);
  }

  /// Parses a region if present. If the region is present, a new region is
  /// allocated and placed in `region`. If no region is present, `region`
  /// remains untouched.
  OptionalParseResult
  parseOptionalRegion(std::unique_ptr<Region> &region,
                      ArrayRef<Argument> arguments,
                      bool enableNameShadowing = false) override {
    if (parser.getToken().isNot(Token::l_brace))
      return llvm::None;
    std::unique_ptr<Region> newRegion = std::make_unique<Region>();
    if (parseRegion(*newRegion, arguments, enableNameShadowing))
      return failure();

    region = std::move(newRegion);
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Successor Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a single operation successor.
  ParseResult parseSuccessor(Block *&dest) override {
    return parser.parseSuccessor(dest);
  }

  /// Parse an optional operation successor and its operand list.
  OptionalParseResult parseOptionalSuccessor(Block *&dest) override {
    if (!parser.getToken().isOrIsCodeCompletionFor(Token::caret_identifier))
      return llvm::None;
    return parseSuccessor(dest);
  }

  /// Parse a single operation successor and its operand list.
  ParseResult
  parseSuccessorAndUseList(Block *&dest,
                           SmallVectorImpl<Value> &operands) override {
    if (parseSuccessor(dest))
      return failure();

    // Handle optional arguments.
    if (succeeded(parseOptionalLParen()) &&
        (parser.parseOptionalSSAUseAndTypeList(operands) || parseRParen())) {
      return failure();
    }
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Type Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a list of assignments of the form
  ///   (%x1 = %y1, %x2 = %y2, ...).
  OptionalParseResult parseOptionalAssignmentList(
      SmallVectorImpl<Argument> &lhs,
      SmallVectorImpl<UnresolvedOperand> &rhs) override {
    if (failed(parseOptionalLParen()))
      return llvm::None;

    auto parseElt = [&]() -> ParseResult {
      if (parseArgument(lhs.emplace_back()) || parseEqual() ||
          parseOperand(rhs.emplace_back()))
        return failure();
      return success();
    };
    return parser.parseCommaSeparatedListUntil(Token::r_paren, parseElt);
  }

  /// Parse a loc(...) specifier if present, filling in result if so.
  ParseResult
  parseOptionalLocationSpecifier(Optional<Location> &result) override {
    // If there is a 'loc' we parse a trailing location.
    if (!parser.consumeIf(Token::kw_loc))
      return success();
    LocationAttr directLoc;
    if (parser.parseToken(Token::l_paren, "expected '(' in location"))
      return failure();

    Token tok = parser.getToken();

    // Check to see if we are parsing a location alias.
    // Otherwise, we parse the location directly.
    if (tok.is(Token::hash_identifier)) {
      if (parser.parseLocationAlias(directLoc))
        return failure();
    } else if (parser.parseLocationInstance(directLoc)) {
      return failure();
    }

    if (parser.parseToken(Token::r_paren, "expected ')' in location"))
      return failure();

    result = directLoc;
    return success();
  }

private:
  /// Information about the result name specifiers.
  ArrayRef<OperationParser::ResultRecord> resultIDs;

  /// The abstract information of the operation.
  function_ref<ParseResult(OpAsmParser &, OperationState &)> parseAssembly;
  bool isIsolatedFromAbove;
  StringRef opName;

  /// The backing operation parser.
  OperationParser &parser;
};
} // namespace

FailureOr<OperationName> OperationParser::parseCustomOperationName() {
  Token nameTok = getToken();
  StringRef opName = nameTok.getSpelling();
  if (opName.empty())
    return (emitError("empty operation name is invalid"), failure());
  consumeToken();

  // Check to see if this operation name is already registered.
  Optional<RegisteredOperationName> opInfo =
      RegisteredOperationName::lookup(opName, getContext());
  if (opInfo)
    return *opInfo;

  // If the operation doesn't have a dialect prefix try using the default
  // dialect.
  auto opNameSplit = opName.split('.');
  StringRef dialectName = opNameSplit.first;
  std::string opNameStorage;
  if (opNameSplit.second.empty()) {
    // If the name didn't have a prefix, check for a code completion request.
    if (getToken().isCodeCompletion() && opName.back() == '.')
      return codeCompleteOperationName(dialectName);

    dialectName = getState().defaultDialectStack.back();
    opNameStorage = (dialectName + "." + opName).str();
    opName = opNameStorage;
  }

  // Try to load the dialect before returning the operation name to make sure
  // the operation has a chance to be registered.
  getContext()->getOrLoadDialect(dialectName);
  return OperationName(opName, getContext());
}

Operation *
OperationParser::parseCustomOperation(ArrayRef<ResultRecord> resultIDs) {
  SMLoc opLoc = getToken().getLoc();
  StringRef originalOpName = getTokenSpelling();

  FailureOr<OperationName> opNameInfo = parseCustomOperationName();
  if (failed(opNameInfo))
    return nullptr;
  StringRef opName = opNameInfo->getStringRef();

  // This is the actual hook for the custom op parsing, usually implemented by
  // the op itself (`Op::parse()`). We retrieve it either from the
  // RegisteredOperationName or from the Dialect.
  function_ref<ParseResult(OpAsmParser &, OperationState &)> parseAssemblyFn;
  bool isIsolatedFromAbove = false;

  StringRef defaultDialect = "";
  if (auto opInfo = opNameInfo->getRegisteredInfo()) {
    parseAssemblyFn = opInfo->getParseAssemblyFn();
    isIsolatedFromAbove = opInfo->hasTrait<OpTrait::IsIsolatedFromAbove>();
    auto *iface = opInfo->getInterface<OpAsmOpInterface>();
    if (iface && !iface->getDefaultDialect().empty())
      defaultDialect = iface->getDefaultDialect();
  } else {
    Optional<Dialect::ParseOpHook> dialectHook;
    if (Dialect *dialect = opNameInfo->getDialect())
      dialectHook = dialect->getParseOperationHook(opName);
    if (!dialectHook) {
      InFlightDiagnostic diag =
          emitError(opLoc) << "custom op '" << originalOpName << "' is unknown";
      if (originalOpName != opName)
        diag << " (tried '" << opName << "' as well)";
      return nullptr;
    }
    parseAssemblyFn = *dialectHook;
  }
  getState().defaultDialectStack.push_back(defaultDialect);
  auto restoreDefaultDialect = llvm::make_scope_exit(
      [&]() { getState().defaultDialectStack.pop_back(); });

  // If the custom op parser crashes, produce some indication to help
  // debugging.
  llvm::PrettyStackTraceFormat fmt("MLIR Parser: custom op parser '%s'",
                                   opNameInfo->getIdentifier().data());

  // Get location information for the operation.
  auto srcLocation = getEncodedSourceLocation(opLoc);
  OperationState opState(srcLocation, *opNameInfo);

  // If we are populating the parser state, start a new operation definition.
  if (state.asmState)
    state.asmState->startOperationDefinition(opState.name);

  // Have the op implementation take a crack and parsing this.
  CleanupOpStateRegions guard{opState};
  CustomOpAsmParser opAsmParser(opLoc, resultIDs, parseAssemblyFn,
                                isIsolatedFromAbove, opName, *this);
  if (opAsmParser.parseOperation(opState))
    return nullptr;

  // If it emitted an error, we failed.
  if (opAsmParser.didEmitError())
    return nullptr;

  // Otherwise, create the operation and try to parse a location for it.
  Operation *op = opBuilder.create(opState);
  if (parseTrailingLocationSpecifier(op))
    return nullptr;
  return op;
}

ParseResult OperationParser::parseLocationAlias(LocationAttr &loc) {
  Token tok = getToken();
  consumeToken(Token::hash_identifier);
  StringRef identifier = tok.getSpelling().drop_front();
  if (identifier.contains('.')) {
    return emitError(tok.getLoc())
           << "expected location, but found dialect attribute: '#" << identifier
           << "'";
  }

  // If this alias can be resolved, do it now.
  Attribute attr = state.symbols.attributeAliasDefinitions.lookup(identifier);
  if (attr) {
    if (!(loc = attr.dyn_cast<LocationAttr>()))
      return emitError(tok.getLoc())
             << "expected location, but found '" << attr << "'";
  } else {
    // Otherwise, remember this operation and resolve its location later.
    // In the meantime, use a special OpaqueLoc as a marker.
    loc = OpaqueLoc::get(deferredLocsReferences.size(),
                         TypeID::get<DeferredLocInfo *>(),
                         UnknownLoc::get(getContext()));
    deferredLocsReferences.push_back(DeferredLocInfo{tok.getLoc(), identifier});
  }
  return success();
}

ParseResult
OperationParser::parseTrailingLocationSpecifier(OpOrArgument opOrArgument) {
  // If there is a 'loc' we parse a trailing location.
  if (!consumeIf(Token::kw_loc))
    return success();
  if (parseToken(Token::l_paren, "expected '(' in location"))
    return failure();
  Token tok = getToken();

  // Check to see if we are parsing a location alias.
  // Otherwise, we parse the location directly.
  LocationAttr directLoc;
  if (tok.is(Token::hash_identifier)) {
    if (parseLocationAlias(directLoc))
      return failure();
  } else if (parseLocationInstance(directLoc)) {
    return failure();
  }

  if (parseToken(Token::r_paren, "expected ')' in location"))
    return failure();

  if (auto *op = opOrArgument.dyn_cast<Operation *>())
    op->setLoc(directLoc);
  else
    opOrArgument.get<BlockArgument>().setLoc(directLoc);
  return success();
}

//===----------------------------------------------------------------------===//
// Region Parsing
//===----------------------------------------------------------------------===//

ParseResult OperationParser::parseRegion(Region &region,
                                         ArrayRef<Argument> entryArguments,
                                         bool isIsolatedNameScope) {
  // Parse the '{'.
  Token lBraceTok = getToken();
  if (parseToken(Token::l_brace, "expected '{' to begin a region"))
    return failure();

  // If we are populating the parser state, start a new region definition.
  if (state.asmState)
    state.asmState->startRegionDefinition();

  // Parse the region body.
  if ((!entryArguments.empty() || getToken().isNot(Token::r_brace)) &&
      parseRegionBody(region, lBraceTok.getLoc(), entryArguments,
                      isIsolatedNameScope)) {
    return failure();
  }
  consumeToken(Token::r_brace);

  // If we are populating the parser state, finalize this region.
  if (state.asmState)
    state.asmState->finalizeRegionDefinition();

  return success();
}

ParseResult OperationParser::parseRegionBody(Region &region, SMLoc startLoc,
                                             ArrayRef<Argument> entryArguments,
                                             bool isIsolatedNameScope) {
  auto currentPt = opBuilder.saveInsertionPoint();

  // Push a new named value scope.
  pushSSANameScope(isIsolatedNameScope);

  // Parse the first block directly to allow for it to be unnamed.
  auto owningBlock = std::make_unique<Block>();
  Block *block = owningBlock.get();

  // If this block is not defined in the source file, add a definition for it
  // now in the assembly state. Blocks with a name will be defined when the name
  // is parsed.
  if (state.asmState && getToken().isNot(Token::caret_identifier))
    state.asmState->addDefinition(block, startLoc);

  // Add arguments to the entry block if we had the form with explicit names.
  if (!entryArguments.empty() && !entryArguments[0].ssaName.name.empty()) {
    // If we had named arguments, then don't allow a block name.
    if (getToken().is(Token::caret_identifier))
      return emitError("invalid block name in region with named arguments");

    for (auto &entryArg : entryArguments) {
      auto &argInfo = entryArg.ssaName;

      // Ensure that the argument was not already defined.
      if (auto defLoc = getReferenceLoc(argInfo.name, argInfo.number)) {
        return emitError(argInfo.location, "region entry argument '" +
                                               argInfo.name +
                                               "' is already in use")
                   .attachNote(getEncodedSourceLocation(*defLoc))
               << "previously referenced here";
      }
      Location loc = entryArg.sourceLoc.has_value()
                         ? entryArg.sourceLoc.value()
                         : getEncodedSourceLocation(argInfo.location);
      BlockArgument arg = block->addArgument(entryArg.type, loc);

      // Add a definition of this arg to the assembly state if provided.
      if (state.asmState)
        state.asmState->addDefinition(arg, argInfo.location);

      // Record the definition for this argument.
      if (addDefinition(argInfo, arg))
        return failure();
    }
  }

  if (parseBlock(block))
    return failure();

  // Verify that no other arguments were parsed.
  if (!entryArguments.empty() &&
      block->getNumArguments() > entryArguments.size()) {
    return emitError("entry block arguments were already defined");
  }

  // Parse the rest of the region.
  region.push_back(owningBlock.release());
  while (getToken().isNot(Token::r_brace)) {
    Block *newBlock = nullptr;
    if (parseBlock(newBlock))
      return failure();
    region.push_back(newBlock);
  }

  // Pop the SSA value scope for this region.
  if (popSSANameScope())
    return failure();

  // Reset the original insertion point.
  opBuilder.restoreInsertionPoint(currentPt);
  return success();
}

//===----------------------------------------------------------------------===//
// Block Parsing
//===----------------------------------------------------------------------===//

/// Block declaration.
///
///   block ::= block-label? operation*
///   block-label    ::= block-id block-arg-list? `:`
///   block-id       ::= caret-id
///   block-arg-list ::= `(` ssa-id-and-type-list? `)`
///
ParseResult OperationParser::parseBlock(Block *&block) {
  // The first block of a region may already exist, if it does the caret
  // identifier is optional.
  if (block && getToken().isNot(Token::caret_identifier))
    return parseBlockBody(block);

  SMLoc nameLoc = getToken().getLoc();
  auto name = getTokenSpelling();
  if (parseToken(Token::caret_identifier, "expected block name"))
    return failure();

  // Define the block with the specified name.
  auto &blockAndLoc = getBlockInfoByName(name);
  blockAndLoc.loc = nameLoc;

  // Use a unique pointer for in-flight block being parsed. Release ownership
  // only in the case of a successful parse. This ensures that the Block
  // allocated is released if the parse fails and control returns early.
  std::unique_ptr<Block> inflightBlock;
  auto cleanupOnFailure = llvm::make_scope_exit([&] {
    if (inflightBlock)
      inflightBlock->dropAllDefinedValueUses();
  });

  // If a block has yet to be set, this is a new definition. If the caller
  // provided a block, use it. Otherwise create a new one.
  if (!blockAndLoc.block) {
    if (block) {
      blockAndLoc.block = block;
    } else {
      inflightBlock = std::make_unique<Block>();
      blockAndLoc.block = inflightBlock.get();
    }

    // Otherwise, the block has a forward declaration. Forward declarations are
    // removed once defined, so if we are defining a existing block and it is
    // not a forward declaration, then it is a redeclaration. Fail if the block
    // was already defined.
  } else if (!eraseForwardRef(blockAndLoc.block)) {
    return emitError(nameLoc, "redefinition of block '") << name << "'";
  } else {
    // This was a forward reference block that is now floating. Keep track of it
    // as inflight in case of error, so that it gets cleaned up properly.
    inflightBlock.reset(blockAndLoc.block);
  }

  // Populate the high level assembly state if necessary.
  if (state.asmState)
    state.asmState->addDefinition(blockAndLoc.block, nameLoc);
  block = blockAndLoc.block;

  // If an argument list is present, parse it.
  if (getToken().is(Token::l_paren))
    if (parseOptionalBlockArgList(block))
      return failure();
  if (parseToken(Token::colon, "expected ':' after block name"))
    return failure();

  // Parse the body of the block.
  ParseResult res = parseBlockBody(block);

  // If parsing was successful, drop the inflight block. We relinquish ownership
  // back up to the caller.
  if (succeeded(res))
    (void)inflightBlock.release();
  return res;
}

ParseResult OperationParser::parseBlockBody(Block *block) {
  // Set the insertion point to the end of the block to parse.
  opBuilder.setInsertionPointToEnd(block);

  // Parse the list of operations that make up the body of the block.
  while (getToken().isNot(Token::caret_identifier, Token::r_brace))
    if (parseOperation())
      return failure();

  return success();
}

/// Get the block with the specified name, creating it if it doesn't already
/// exist.  The location specified is the point of use, which allows
/// us to diagnose references to blocks that are not defined precisely.
Block *OperationParser::getBlockNamed(StringRef name, SMLoc loc) {
  BlockDefinition &blockDef = getBlockInfoByName(name);
  if (!blockDef.block) {
    blockDef = {new Block(), loc};
    insertForwardRef(blockDef.block, blockDef.loc);
  }

  // Populate the high level assembly state if necessary.
  if (state.asmState)
    state.asmState->addUses(blockDef.block, loc);

  return blockDef.block;
}

/// Parse a (possibly empty) list of SSA operands with types as block arguments
/// enclosed in parentheses.
///
///   value-id-and-type-list ::= value-id-and-type (`,` ssa-id-and-type)*
///   block-arg-list ::= `(` value-id-and-type-list? `)`
///
ParseResult OperationParser::parseOptionalBlockArgList(Block *owner) {
  if (getToken().is(Token::r_brace))
    return success();

  // If the block already has arguments, then we're handling the entry block.
  // Parse and register the names for the arguments, but do not add them.
  bool definingExistingArgs = owner->getNumArguments() != 0;
  unsigned nextArgument = 0;

  return parseCommaSeparatedList(Delimiter::Paren, [&]() -> ParseResult {
    return parseSSADefOrUseAndType(
        [&](UnresolvedOperand useInfo, Type type) -> ParseResult {
          BlockArgument arg;

          // If we are defining existing arguments, ensure that the argument
          // has already been created with the right type.
          if (definingExistingArgs) {
            // Otherwise, ensure that this argument has already been created.
            if (nextArgument >= owner->getNumArguments())
              return emitError("too many arguments specified in argument list");

            // Finally, make sure the existing argument has the correct type.
            arg = owner->getArgument(nextArgument++);
            if (arg.getType() != type)
              return emitError("argument and block argument type mismatch");
          } else {
            auto loc = getEncodedSourceLocation(useInfo.location);
            arg = owner->addArgument(type, loc);
          }

          // If the argument has an explicit loc(...) specifier, parse and apply
          // it.
          if (parseTrailingLocationSpecifier(arg))
            return failure();

          // Mark this block argument definition in the parser state if it was
          // provided.
          if (state.asmState)
            state.asmState->addDefinition(arg, useInfo.location);

          return addDefinition(useInfo, arg);
        });
  });
}

//===----------------------------------------------------------------------===//
// Code Completion
//===----------------------------------------------------------------------===//

ParseResult OperationParser::codeCompleteSSAUse() {
  std::string detailData;
  llvm::raw_string_ostream detailOS(detailData);
  for (IsolatedSSANameScope &scope : isolatedNameScopes) {
    for (auto &it : scope.values) {
      if (it.second.empty())
        continue;
      Value frontValue = it.second.front().value;

      // If the value isn't a forward reference, we also add the name of the op
      // to the detail.
      if (auto result = frontValue.dyn_cast<OpResult>()) {
        if (!forwardRefPlaceholders.count(result))
          detailOS << result.getOwner()->getName() << ": ";
      } else {
        detailOS << "arg #" << frontValue.cast<BlockArgument>().getArgNumber()
                 << ": ";
      }

      // Emit the type of the values to aid with completion selection.
      detailOS << frontValue.getType();

      // FIXME: We should define a policy for packed values, e.g. with a limit
      // on the detail size, but it isn't clear what would be useful right now.
      // For now we just only emit the first type.
      if (it.second.size() > 1)
        detailOS << ", ...";

      state.codeCompleteContext->appendSSAValueCompletion(
          it.getKey(), std::move(detailOS.str()));
    }
  }

  return failure();
}

ParseResult OperationParser::codeCompleteBlock() {
  // Don't provide completions if the token isn't empty, e.g. this avoids
  // weirdness when we encounter a `.` within the identifier.
  StringRef spelling = getTokenSpelling();
  if (!(spelling.empty() || spelling == "^"))
    return failure();

  for (const auto &it : blocksByName.back())
    state.codeCompleteContext->appendBlockCompletion(it.getFirst());
  return failure();
}

//===----------------------------------------------------------------------===//
// Top-level entity parsing.
//===----------------------------------------------------------------------===//

namespace {
/// This parser handles entities that are only valid at the top level of the
/// file.
class TopLevelOperationParser : public Parser {
public:
  explicit TopLevelOperationParser(ParserState &state) : Parser(state) {}

  /// Parse a set of operations into the end of the given Block.
  ParseResult parse(Block *topLevelBlock, Location parserLoc);

private:
  /// Parse an attribute alias declaration.
  ///
  ///   attribute-alias-def ::= '#' alias-name `=` attribute-value
  ///
  ParseResult parseAttributeAliasDef();

  /// Parse a type alias declaration.
  ///
  ///   type-alias-def ::= '!' alias-name `=` type
  ///
  ParseResult parseTypeAliasDef();

  /// Parse a top-level file metadata dictionary.
  ///
  ///   file-metadata-dict ::= '{-#' file-metadata-entry* `#-}'
  ///
  ParseResult parseFileMetadataDictionary();

  /// Parse a resource metadata dictionary.
  ParseResult parseResourceFileMetadata(
      function_ref<ParseResult(StringRef, SMLoc)> parseBody);
  ParseResult parseDialectResourceFileMetadata();
  ParseResult parseExternalResourceFileMetadata();
};

/// This class represents an implementation of a resource entry for the MLIR
/// textual format.
class ParsedResourceEntry : public AsmParsedResourceEntry {
public:
  ParsedResourceEntry(StringRef key, SMLoc keyLoc, Token value, Parser &p)
      : key(key), keyLoc(keyLoc), value(value), p(p) {}
  ~ParsedResourceEntry() override = default;

  StringRef getKey() const final { return key; }

  InFlightDiagnostic emitError() const final { return p.emitError(keyLoc); }

  FailureOr<bool> parseAsBool() const final {
    if (value.is(Token::kw_true))
      return true;
    if (value.is(Token::kw_false))
      return false;
    return p.emitError(value.getLoc(),
                       "expected 'true' or 'false' value for key '" + key +
                           "'");
  }

  FailureOr<std::string> parseAsString() const final {
    if (value.isNot(Token::string))
      return p.emitError(value.getLoc(),
                         "expected string value for key '" + key + "'");
    return value.getStringValue();
  }

  FailureOr<AsmResourceBlob>
  parseAsBlob(BlobAllocatorFn allocator) const final {
    // Blob data within then textual format is represented as a hex string.
    // TODO: We could avoid an additional alloc+copy here if we pre-allocated
    // the buffer to use during hex processing.
    Optional<std::string> blobData =
        value.is(Token::string) ? value.getHexStringValue() : llvm::None;
    if (!blobData)
      return p.emitError(value.getLoc(),
                         "expected hex string blob for key '" + key + "'");

    // Extract the alignment of the blob data, which gets stored at the
    // beginning of the string.
    if (blobData->size() < sizeof(uint32_t)) {
      return p.emitError(value.getLoc(),
                         "expected hex string blob for key '" + key +
                             "' to encode alignment in first 4 bytes");
    }
    llvm::support::ulittle32_t align;
    memcpy(&align, blobData->data(), sizeof(uint32_t));

    // Get the data portion of the blob.
    StringRef data = StringRef(*blobData).drop_front(sizeof(uint32_t));
    if (data.empty())
      return AsmResourceBlob();

    // Allocate memory for the blob using the provided allocator and copy the
    // data into it.
    AsmResourceBlob blob = allocator(data.size(), align);
    assert(llvm::isAddrAligned(llvm::Align(align), blob.getData().data()) &&
           blob.isMutable() &&
           "blob allocator did not return a properly aligned address");
    memcpy(blob.getMutableData().data(), data.data(), data.size());
    return blob;
  }

private:
  StringRef key;
  SMLoc keyLoc;
  Token value;
  Parser &p;
};
} // namespace

ParseResult TopLevelOperationParser::parseAttributeAliasDef() {
  assert(getToken().is(Token::hash_identifier));
  StringRef aliasName = getTokenSpelling().drop_front();

  // Check for redefinitions.
  if (state.symbols.attributeAliasDefinitions.count(aliasName) > 0)
    return emitError("redefinition of attribute alias id '" + aliasName + "'");

  // Make sure this isn't invading the dialect attribute namespace.
  if (aliasName.contains('.'))
    return emitError("attribute names with a '.' are reserved for "
                     "dialect-defined names");

  consumeToken(Token::hash_identifier);

  // Parse the '='.
  if (parseToken(Token::equal, "expected '=' in attribute alias definition"))
    return failure();

  // Parse the attribute value.
  Attribute attr = parseAttribute();
  if (!attr)
    return failure();

  state.symbols.attributeAliasDefinitions[aliasName] = attr;
  return success();
}

ParseResult TopLevelOperationParser::parseTypeAliasDef() {
  assert(getToken().is(Token::exclamation_identifier));
  StringRef aliasName = getTokenSpelling().drop_front();

  // Check for redefinitions.
  if (state.symbols.typeAliasDefinitions.count(aliasName) > 0)
    return emitError("redefinition of type alias id '" + aliasName + "'");

  // Make sure this isn't invading the dialect type namespace.
  if (aliasName.contains('.'))
    return emitError("type names with a '.' are reserved for "
                     "dialect-defined names");
  consumeToken(Token::exclamation_identifier);

  // Parse the '='.
  if (parseToken(Token::equal, "expected '=' in type alias definition"))
    return failure();

  // Parse the type.
  Type aliasedType = parseType();
  if (!aliasedType)
    return failure();

  // Register this alias with the parser state.
  state.symbols.typeAliasDefinitions.try_emplace(aliasName, aliasedType);
  return success();
}

ParseResult TopLevelOperationParser::parseFileMetadataDictionary() {
  consumeToken(Token::file_metadata_begin);
  return parseCommaSeparatedListUntil(
      Token::file_metadata_end, [&]() -> ParseResult {
        // Parse the key of the metadata dictionary.
        SMLoc keyLoc = getToken().getLoc();
        StringRef key;
        if (failed(parseOptionalKeyword(&key)))
          return emitError("expected identifier key in file "
                           "metadata dictionary");
        if (parseToken(Token::colon, "expected ':'"))
          return failure();

        // Process the metadata entry.
        if (key == "dialect_resources")
          return parseDialectResourceFileMetadata();
        if (key == "external_resources")
          return parseExternalResourceFileMetadata();
        return emitError(keyLoc, "unknown key '" + key +
                                     "' in file metadata dictionary");
      });
}

ParseResult TopLevelOperationParser::parseResourceFileMetadata(
    function_ref<ParseResult(StringRef, SMLoc)> parseBody) {
  if (parseToken(Token::l_brace, "expected '{'"))
    return failure();

  return parseCommaSeparatedListUntil(Token::r_brace, [&]() -> ParseResult {
    // Parse the top-level name entry.
    SMLoc nameLoc = getToken().getLoc();
    StringRef name;
    if (failed(parseOptionalKeyword(&name)))
      return emitError("expected identifier key for 'resource' entry");

    if (parseToken(Token::colon, "expected ':'") ||
        parseToken(Token::l_brace, "expected '{'"))
      return failure();
    return parseBody(name, nameLoc);
  });
}

ParseResult TopLevelOperationParser::parseDialectResourceFileMetadata() {
  return parseResourceFileMetadata([&](StringRef name,
                                       SMLoc nameLoc) -> ParseResult {
    // Lookup the dialect and check that it can handle a resource entry.
    Dialect *dialect = getContext()->getOrLoadDialect(name);
    if (!dialect)
      return emitError(nameLoc, "dialect '" + name + "' is unknown");
    const auto *handler = dyn_cast<OpAsmDialectInterface>(dialect);
    if (!handler) {
      return emitError() << "unexpected 'resource' section for dialect '"
                         << dialect->getNamespace() << "'";
    }

    return parseCommaSeparatedListUntil(Token::r_brace, [&]() -> ParseResult {
      // Parse the name of the resource entry.
      SMLoc keyLoc = getToken().getLoc();
      StringRef key;
      if (failed(parseResourceHandle(handler, key)) ||
          parseToken(Token::colon, "expected ':'"))
        return failure();
      Token valueTok = getToken();
      consumeToken();

      ParsedResourceEntry entry(key, keyLoc, valueTok, *this);
      return handler->parseResource(entry);
    });
  });
}

ParseResult TopLevelOperationParser::parseExternalResourceFileMetadata() {
  return parseResourceFileMetadata([&](StringRef name,
                                       SMLoc nameLoc) -> ParseResult {
    AsmResourceParser *handler = state.config.getResourceParser(name);

    // TODO: Should we require handling external resources in some scenarios?
    if (!handler) {
      emitWarning(getEncodedSourceLocation(nameLoc))
          << "ignoring unknown external resources for '" << name << "'";
    }

    return parseCommaSeparatedListUntil(Token::r_brace, [&]() -> ParseResult {
      // Parse the name of the resource entry.
      SMLoc keyLoc = getToken().getLoc();
      StringRef key;
      if (failed(parseOptionalKeyword(&key)))
        return emitError(
            "expected identifier key for 'external_resources' entry");
      if (parseToken(Token::colon, "expected ':'"))
        return failure();
      Token valueTok = getToken();
      consumeToken();

      if (!handler)
        return success();
      ParsedResourceEntry entry(key, keyLoc, valueTok, *this);
      return handler->parseResource(entry);
    });
  });
}

ParseResult TopLevelOperationParser::parse(Block *topLevelBlock,
                                           Location parserLoc) {
  // Create a top-level operation to contain the parsed state.
  OwningOpRef<ModuleOp> topLevelOp(ModuleOp::create(parserLoc));
  OperationParser opParser(state, topLevelOp.get());
  while (true) {
    switch (getToken().getKind()) {
    default:
      // Parse a top-level operation.
      if (opParser.parseOperation())
        return failure();
      break;

    // If we got to the end of the file, then we're done.
    case Token::eof: {
      if (opParser.finalize())
        return failure();

      // Splice the blocks of the parsed operation over to the provided
      // top-level block.
      auto &parsedOps = topLevelOp->getBody()->getOperations();
      auto &destOps = topLevelBlock->getOperations();
      destOps.splice(destOps.empty() ? destOps.end() : std::prev(destOps.end()),
                     parsedOps, parsedOps.begin(), parsedOps.end());
      return success();
    }

    // If we got an error token, then the lexer already emitted an error, just
    // stop.  Someday we could introduce error recovery if there was demand
    // for it.
    case Token::error:
      return failure();

    // Parse an attribute alias.
    case Token::hash_identifier:
      if (parseAttributeAliasDef())
        return failure();
      break;

    // Parse a type alias.
    case Token::exclamation_identifier:
      if (parseTypeAliasDef())
        return failure();
      break;

      // Parse a file-level metadata dictionary.
    case Token::file_metadata_begin:
      if (parseFileMetadataDictionary())
        return failure();
      break;
    }
  }
}

//===----------------------------------------------------------------------===//

LogicalResult
mlir::parseAsmSourceFile(const llvm::SourceMgr &sourceMgr, Block *block,
                         const ParserConfig &config, AsmParserState *asmState,
                         AsmParserCodeCompleteContext *codeCompleteContext) {
  const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  Location parserLoc =
      FileLineColLoc::get(config.getContext(), sourceBuf->getBufferIdentifier(),
                          /*line=*/0, /*column=*/0);

  SymbolState aliasState;
  ParserState state(sourceMgr, config, aliasState, asmState,
                    codeCompleteContext);
  return TopLevelOperationParser(state).parse(block, parserLoc);
}
