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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::detail;
using llvm::MemoryBuffer;
using llvm::SMLoc;
using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// Parse a comma separated list of elements that must have at least one entry
/// in it.
ParseResult
Parser::parseCommaSeparatedList(function_ref<ParseResult()> parseElement) {
  // Non-empty case starts with an element.
  if (parseElement())
    return failure();

  // Otherwise we have a list of comma separated elements.
  while (consumeIf(Token::comma)) {
    if (parseElement())
      return failure();
  }
  return success();
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
      return emitError("expected list element");
    consumeToken(rightToken);
    return success();
  }

  if (parseCommaSeparatedList(parseElement) ||
      parseToken(rightToken, "expected ',' or '" +
                                 Token::getTokenSpelling(rightToken) + "'"))
    return failure();

  return success();
}

InFlightDiagnostic Parser::emitError(SMLoc loc, const Twine &message) {
  auto diag = mlir::emitError(getEncodedSourceLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(Token::error))
    diag.abandon();
  return diag;
}

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult Parser::parseToken(Token::Kind expectedToken,
                               const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
}

/// Parse an optional integer value from the stream.
OptionalParseResult Parser::parseOptionalInteger(uint64_t &result) {
  Token curToken = getToken();
  if (curToken.isNot(Token::integer, Token::minus))
    return llvm::None;

  bool negative = consumeIf(Token::minus);
  Token curTok = getToken();
  if (parseToken(Token::integer, "expected integer value"))
    return failure();

  auto val = curTok.getUInt64IntegerValue();
  if (!val)
    return emitError(curTok.getLoc(), "integer value too large");
  result = negative ? -*val : *val;
  return success();
}

//===----------------------------------------------------------------------===//
// OperationParser
//===----------------------------------------------------------------------===//

namespace {
/// This class provides support for parsing operations and regions of
/// operations.
class OperationParser : public Parser {
public:
  OperationParser(ParserState &state, Operation *topLevelOp)
      : Parser(state), opBuilder(topLevelOp->getRegion(0)),
        topLevelOp(topLevelOp) {
    // The top level operation starts a new name scope.
    pushSSANameScope(/*isIsolated=*/true);
  }

  ~OperationParser();

  /// After parsing is finished, this function must be called to see if there
  /// are any remaining issues.
  ParseResult finalize();

  //===--------------------------------------------------------------------===//
  // SSA Value Handling
  //===--------------------------------------------------------------------===//

  /// This represents a use of an SSA value in the program.  The first two
  /// entries in the tuple are the name and result number of a reference.  The
  /// third is the location of the reference, which is used in case this ends
  /// up being a use of an undefined value.
  struct SSAUseInfo {
    StringRef name;  // Value name, e.g. %42 or %abc
    unsigned number; // Number, specified with #12
    SMLoc loc;       // Location of first definition or use.
  };

  /// Push a new SSA name scope to the parser.
  void pushSSANameScope(bool isIsolated);

  /// Pop the last SSA name scope from the parser.
  ParseResult popSSANameScope();

  /// Register a definition of a value with the symbol table.
  ParseResult addDefinition(SSAUseInfo useInfo, Value value);

  /// Parse an optional list of SSA uses into 'results'.
  ParseResult parseOptionalSSAUseList(SmallVectorImpl<SSAUseInfo> &results);

  /// Parse a single SSA use into 'result'.
  ParseResult parseSSAUse(SSAUseInfo &result);

  /// Given a reference to an SSA value and its type, return a reference. This
  /// returns null on failure.
  Value resolveSSAUse(SSAUseInfo useInfo, Type type);

  ParseResult
  parseSSADefOrUseAndType(function_ref<ParseResult(SSAUseInfo, Type)> action);

  ParseResult parseOptionalSSAUseAndTypeList(SmallVectorImpl<Value> &results);

  /// Return the location of the value identified by its name and number if it
  /// has been already reference.
  Optional<SMLoc> getReferenceLoc(StringRef name, unsigned number) {
    auto &values = isolatedNameScopes.back().values;
    if (!values.count(name) || number >= values[name].size())
      return {};
    if (values[name][number].first)
      return values[name][number].second;
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

  /// Parse an operation instance that is in the generic form and insert it at
  /// the provided insertion point.
  Operation *parseGenericOperation(Block *insertBlock,
                                   Block::iterator insertPt);

  /// Parse an optional trailing location for the given operation.
  ///
  ///   trailing-location ::= (`loc` (`(` location `)` | attribute-alias))?
  ///
  ParseResult parseTrailingOperationLocation(Operation *op);

  /// This is the structure of a result specifier in the assembly syntax,
  /// including the name, number of results, and location.
  using ResultRecord = std::tuple<StringRef, unsigned, SMLoc>;

  /// Parse an operation instance that is in the op-defined custom form.
  /// resultInfo specifies information about the "%name =" specifiers.
  Operation *parseCustomOperation(ArrayRef<ResultRecord> resultIDs);

  //===--------------------------------------------------------------------===//
  // Region Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a region into 'region' with the provided entry block arguments.
  /// 'isIsolatedNameScope' indicates if the naming scope of this region is
  /// isolated from those above.
  ParseResult parseRegion(Region &region,
                          ArrayRef<std::pair<SSAUseInfo, Type>> entryArguments,
                          bool isIsolatedNameScope = false);

  /// Parse a region body into 'region'.
  ParseResult parseRegionBody(Region &region);

  //===--------------------------------------------------------------------===//
  // Block Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a new block into 'block'.
  ParseResult parseBlock(Block *&block);

  /// Parse a list of operations into 'block'.
  ParseResult parseBlockBody(Block *block);

  /// Parse a (possibly empty) list of block arguments.
  ParseResult parseOptionalBlockArgList(SmallVectorImpl<BlockArgument> &results,
                                        Block *owner);

  /// Get the block with the specified name, creating it if it doesn't
  /// already exist.  The location specified is the point of use, which allows
  /// us to diagnose references to blocks that are not defined precisely.
  Block *getBlockNamed(StringRef name, SMLoc loc);

  /// Define the block with the specified name. Returns the Block* or nullptr in
  /// the case of redefinition.
  Block *defineBlockNamed(StringRef name, SMLoc loc, Block *existing);

private:
  /// Returns the info for a block at the current scope for the given name.
  std::pair<Block *, SMLoc> &getBlockInfoByName(StringRef name) {
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
  SmallVectorImpl<std::pair<Value, SMLoc>> &getSSAValueEntry(StringRef name);

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
    llvm::StringMap<SmallVector<std::pair<Value, SMLoc>, 1>> values;

    /// This keeps track of all of the values defined by a specific name scope.
    SmallVector<llvm::StringSet<>, 2> definitionsPerScope;
  };

  /// A list of isolated name scopes.
  SmallVector<IsolatedSSANameScope, 2> isolatedNameScopes;

  /// This keeps track of the block names as well as the location of the first
  /// reference for each nested name scope. This is used to diagnose invalid
  /// block references and memorize them.
  SmallVector<DenseMap<StringRef, std::pair<Block *, SMLoc>>, 2> blocksByName;
  SmallVector<DenseMap<Block *, SMLoc>, 2> forwardRef;

  /// These are all of the placeholders we've made along with the location of
  /// their first reference, to allow checking for use of undefined values.
  DenseMap<Value, SMLoc> forwardRefPlaceholders;

  /// A set of operations whose locations reference aliases that have yet to
  /// be resolved.
  SmallVector<std::pair<Operation *, Token>, 8> opsWithDeferredLocs;

  /// The builder used when creating parsed operation instances.
  OpBuilder opBuilder;

  /// The top level operation that holds all of the parsed operations.
  Operation *topLevelOp;
};
} // end anonymous namespace

OperationParser::~OperationParser() {
  for (auto &fwd : forwardRefPlaceholders) {
    // Drop all uses of undefined forward declared reference and destroy
    // defining operation.
    fwd.first.dropAllUses();
    fwd.first.getDefiningOp()->destroy();
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

    for (auto entry : errors) {
      auto loc = SMLoc::getFromPointer(entry);
      emitError(loc, "use of undeclared SSA value name");
    }
    return failure();
  }

  // Resolve the locations of any deferred operations.
  auto &attributeAliases = getState().symbols.attributeAliasDefinitions;
  for (std::pair<Operation *, Token> &it : opsWithDeferredLocs) {
    llvm::SMLoc tokLoc = it.second.getLoc();
    StringRef identifier = it.second.getSpelling().drop_front();
    Attribute attr = attributeAliases.lookup(identifier);
    if (!attr)
      return emitError(tokLoc) << "operation location alias was never defined";

    LocationAttr locAttr = attr.dyn_cast<LocationAttr>();
    if (!locAttr)
      return emitError(tokLoc)
             << "expected location, but found '" << attr << "'";
    it.first->setLoc(locAttr);
  }

  // Pop the top level name scope.
  return popSSANameScope();
}

//===----------------------------------------------------------------------===//
// SSA Value Handling
//===----------------------------------------------------------------------===//

void OperationParser::pushSSANameScope(bool isIsolated) {
  blocksByName.push_back(DenseMap<StringRef, std::pair<Block *, SMLoc>>());
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
ParseResult OperationParser::addDefinition(SSAUseInfo useInfo, Value value) {
  auto &entries = getSSAValueEntry(useInfo.name);

  // Make sure there is a slot for this value.
  if (entries.size() <= useInfo.number)
    entries.resize(useInfo.number + 1);

  // If we already have an entry for this, check to see if it was a definition
  // or a forward reference.
  if (auto existing = entries[useInfo.number].first) {
    if (!isForwardRefPlaceholder(existing)) {
      return emitError(useInfo.loc)
          .append("redefinition of SSA value '", useInfo.name, "'")
          .attachNote(getEncodedSourceLocation(entries[useInfo.number].second))
          .append("previously defined here");
    }

    if (existing.getType() != value.getType()) {
      return emitError(useInfo.loc)
          .append("definition of SSA value '", useInfo.name, "#",
                  useInfo.number, "' has type ", value.getType())
          .attachNote(getEncodedSourceLocation(entries[useInfo.number].second))
          .append("previously used here with type ", existing.getType());
    }

    // If it was a forward reference, update everything that used it to use
    // the actual definition instead, delete the forward ref, and remove it
    // from our set of forward references we track.
    existing.replaceAllUsesWith(value);
    existing.getDefiningOp()->destroy();
    forwardRefPlaceholders.erase(existing);
  }

  /// Record this definition for the current scope.
  entries[useInfo.number] = {value, useInfo.loc};
  recordDefinition(useInfo.name);
  return success();
}

/// Parse a (possibly empty) list of SSA operands.
///
///   ssa-use-list ::= ssa-use (`,` ssa-use)*
///   ssa-use-list-opt ::= ssa-use-list?
///
ParseResult
OperationParser::parseOptionalSSAUseList(SmallVectorImpl<SSAUseInfo> &results) {
  if (getToken().isNot(Token::percent_identifier))
    return success();
  return parseCommaSeparatedList([&]() -> ParseResult {
    SSAUseInfo result;
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
ParseResult OperationParser::parseSSAUse(SSAUseInfo &result) {
  result.name = getTokenSpelling();
  result.number = 0;
  result.loc = getToken().getLoc();
  if (parseToken(Token::percent_identifier, "expected SSA operand"))
    return failure();

  // If we have an attribute ID, it is a result number.
  if (getToken().is(Token::hash_identifier)) {
    if (auto value = getToken().getHashIdentifierNumber())
      result.number = value.getValue();
    else
      return emitError("invalid SSA value result number");
    consumeToken(Token::hash_identifier);
  }

  return success();
}

/// Given an unbound reference to an SSA value and its type, return the value
/// it specifies.  This returns null on failure.
Value OperationParser::resolveSSAUse(SSAUseInfo useInfo, Type type) {
  auto &entries = getSSAValueEntry(useInfo.name);

  // If we have already seen a value of this name, return it.
  if (useInfo.number < entries.size() && entries[useInfo.number].first) {
    auto result = entries[useInfo.number].first;
    // Check that the type matches the other uses.
    if (result.getType() == type)
      return result;

    emitError(useInfo.loc, "use of value '")
        .append(useInfo.name,
                "' expects different type than prior uses: ", type, " vs ",
                result.getType())
        .attachNote(getEncodedSourceLocation(entries[useInfo.number].second))
        .append("prior use here");
    return nullptr;
  }

  // Make sure we have enough slots for this.
  if (entries.size() <= useInfo.number)
    entries.resize(useInfo.number + 1);

  // If the value has already been defined and this is an overly large result
  // number, diagnose that.
  if (entries[0].first && !isForwardRefPlaceholder(entries[0].first))
    return (emitError(useInfo.loc, "reference to invalid result number"),
            nullptr);

  // Otherwise, this is a forward reference.  Create a placeholder and remember
  // that we did so.
  auto result = createForwardRefPlaceholder(useInfo.loc, type);
  entries[useInfo.number].first = result;
  entries[useInfo.number].second = useInfo.loc;
  return result;
}

/// Parse an SSA use with an associated type.
///
///   ssa-use-and-type ::= ssa-use `:` type
ParseResult OperationParser::parseSSADefOrUseAndType(
    function_ref<ParseResult(SSAUseInfo, Type)> action) {
  SSAUseInfo useInfo;
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
  SmallVector<SSAUseInfo, 4> valueIDs;
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
SmallVectorImpl<std::pair<Value, SMLoc>> &
OperationParser::getSSAValueEntry(StringRef name) {
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
  auto name = OperationName("placeholder", getContext());
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
      if (!getToken().is(Token::percent_identifier))
        return emitError("expected valid ssa identifier");

      Token nameTok = getToken();
      consumeToken(Token::percent_identifier);

      // If the next token is a ':', we parse the expected result count.
      size_t expectedSubResults = 1;
      if (consumeIf(Token::colon)) {
        // Check that the next token is an integer.
        if (!getToken().is(Token::integer))
          return emitError("expected integer number of results");

        // Check that number of results is > 0.
        auto val = getToken().getUInt64IntegerValue();
        if (!val.hasValue() || val.getValue() < 1)
          return emitError("expected named operation to have atleast 1 result");
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
  if (getToken().is(Token::bare_identifier) || getToken().isKeyword())
    op = parseCustomOperation(resultIDs);
  else if (getToken().is(Token::string))
    op = parseGenericOperation();
  else
    return emitError("expected operation name in quotes");

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

    // Add definitions for each of the result groups.
    unsigned opResI = 0;
    for (ResultRecord &resIt : resultIDs) {
      for (unsigned subRes : llvm::seq<unsigned>(0, std::get<1>(resIt))) {
        if (addDefinition({std::get<0>(resIt), subRes, std::get<2>(resIt)},
                          op->getResult(opResI++)))
          return failure();
      }
    }
  }

  return success();
}

/// Parse a single operation successor.
///
///   successor ::= block-id
///
ParseResult OperationParser::parseSuccessor(Block *&dest) {
  // Verify branch is identifier and get the matching block.
  if (!getToken().is(Token::caret_identifier))
    return emitError("expected block name");
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

  // Lazy load dialects in the context as needed.
  if (!result.name.getAbstractOperation()) {
    StringRef dialectName = StringRef(name).split('.').first;
    if (!getContext()->getLoadedDialect(dialectName) &&
        getContext()->getOrLoadDialect(dialectName)) {
      result.name = OperationName(name, getContext());
    }
  }

  // Parse the operand list.
  SmallVector<SSAUseInfo, 8> operandInfos;
  if (parseToken(Token::l_paren, "expected '(' to start operand list") ||
      parseOptionalSSAUseList(operandInfos) ||
      parseToken(Token::r_paren, "expected ')' to end operand list")) {
    return nullptr;
  }

  // Parse the successor list.
  if (getToken().is(Token::l_square)) {
    // Check if the operation is a known terminator.
    const AbstractOperation *abstractOp = result.name.getAbstractOperation();
    if (abstractOp && !abstractOp->hasTrait<OpTrait::IsTerminator>())
      return emitError("successors in non-terminator"), nullptr;

    SmallVector<Block *, 2> successors;
    if (parseSuccessors(successors))
      return nullptr;
    result.addSuccessors(successors);
  }

  // Parse the region list.
  CleanupOpStateRegions guard{result};
  if (consumeIf(Token::l_paren)) {
    do {
      // Create temporary regions with the top level region as parent.
      result.regions.emplace_back(new Region(topLevelOp));
      if (parseRegion(*result.regions.back(), /*entryArguments=*/{}))
        return nullptr;
    } while (consumeIf(Token::comma));
    if (parseToken(Token::r_paren, "expected ')' to end region list"))
      return nullptr;
  }

  if (getToken().is(Token::l_brace)) {
    if (parseAttributeDict(result.attributes))
      return nullptr;
  }

  if (parseToken(Token::colon, "expected ':' followed by operation type"))
    return nullptr;

  auto typeLoc = getToken().getLoc();
  auto type = parseType();
  if (!type)
    return nullptr;
  auto fnType = type.dyn_cast<FunctionType>();
  if (!fnType)
    return (emitError(typeLoc, "expected function type"), nullptr);

  result.addTypes(fnType.getResults());

  // Check that we have the right number of types for the operands.
  auto operandTypes = fnType.getInputs();
  if (operandTypes.size() != operandInfos.size()) {
    auto plural = "s"[operandInfos.size() == 1];
    return (emitError(typeLoc, "expected ")
                << operandInfos.size() << " operand type" << plural
                << " but had " << operandTypes.size(),
            nullptr);
  }

  // Resolve all of the operands.
  for (unsigned i = 0, e = operandInfos.size(); i != e; ++i) {
    result.operands.push_back(resolveSSAUse(operandInfos[i], operandTypes[i]));
    if (!result.operands.back())
      return nullptr;
  }

  // Create the operation and try to parse a location for it.
  Operation *op = opBuilder.createOperation(result);
  if (parseTrailingOperationLocation(op))
    return nullptr;
  return op;
}

Operation *OperationParser::parseGenericOperation(Block *insertBlock,
                                                  Block::iterator insertPt) {
  OpBuilder::InsertionGuard restoreInsertionPoint(opBuilder);
  opBuilder.setInsertionPoint(insertBlock, insertPt);
  return parseGenericOperation();
}

namespace {
class CustomOpAsmParser : public OpAsmParser {
public:
  CustomOpAsmParser(SMLoc nameLoc,
                    ArrayRef<OperationParser::ResultRecord> resultIDs,
                    const AbstractOperation *opDefinition,
                    OperationParser &parser)
      : nameLoc(nameLoc), resultIDs(resultIDs), opDefinition(opDefinition),
        parser(parser) {}

  /// Parse an instance of the operation described by 'opDefinition' into the
  /// provided operation state.
  ParseResult parseOperation(OperationState &opState) {
    if (opDefinition->parseAssembly(*this, opState))
      return failure();
    // Verify that the parsed attributes does not have duplicate attributes.
    // This can happen if an attribute set during parsing is also specified in
    // the attribute dictionary in the assembly, or the attribute is set
    // multiple during parsing.
    Optional<NamedAttribute> duplicate = opState.attributes.findDuplicate();
    if (duplicate)
      return emitError(getNameLoc(), "attribute '")
             << duplicate->first
             << "' occurs more than once in the attribute list";
    return success();
  }

  Operation *parseGenericOperation(Block *insertBlock,
                                   Block::iterator insertPt) final {
    return parser.parseGenericOperation(insertBlock, insertPt);
  }

  //===--------------------------------------------------------------------===//
  // Utilities
  //===--------------------------------------------------------------------===//

  /// Return if any errors were emitted during parsing.
  bool didEmitError() const { return emittedError; }

  /// Emit a diagnostic at the specified location and return failure.
  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) override {
    emittedError = true;
    return parser.emitError(loc, "custom op '" + opDefinition->name.strref() +
                                     "' " + message);
  }

  llvm::SMLoc getCurrentLocation() override {
    return parser.getToken().getLoc();
  }

  Builder &getBuilder() const override { return parser.builder; }

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
    for (unsigned nameID = 0, e = resultIDs.size(); nameID != e; ++nameID) {
      const auto &entry = resultIDs[nameID];
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

  llvm::SMLoc getNameLoc() const override { return nameLoc; }

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

  /// Parse a '<' token if present.
  ParseResult parseOptionalLess() override {
    return success(parser.consumeIf(Token::less));
  }

  /// Parse a '>' token.
  ParseResult parseGreater() override {
    return parser.parseToken(Token::greater, "expected '>'");
  }

  /// Parse a '>' token if present.
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

  /// Parses a '?' token if present.
  ParseResult parseOptionalQuestion() override {
    return success(parser.consumeIf(Token::question));
  }

  /// Parses a '+' token.
  ParseResult parsePlus() override {
    return parser.parseToken(Token::plus, "expected '+'");
  }

  /// Parses a '+' token if present.
  ParseResult parseOptionalPlus() override {
    return success(parser.consumeIf(Token::plus));
  }

  /// Parses a '*' token.
  ParseResult parseStar() override {
    return parser.parseToken(Token::star, "expected '*'");
  }

  /// Parses a '*' token if present.
  ParseResult parseOptionalStar() override {
    return success(parser.consumeIf(Token::star));
  }

  /// Parse an optional integer value from the stream.
  OptionalParseResult parseOptionalInteger(uint64_t &result) override {
    return parser.parseOptionalInteger(result);
  }

  //===--------------------------------------------------------------------===//
  // Attribute Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an arbitrary attribute of a given type and return it in result.
  ParseResult parseAttribute(Attribute &result, Type type) override {
    result = parser.parseAttribute(type);
    return success(static_cast<bool>(result));
  }

  /// Parse an optional attribute.
  template <typename AttrT>
  OptionalParseResult
  parseOptionalAttributeAndAddToList(AttrT &result, Type type,
                                     StringRef attrName, NamedAttrList &attrs) {
    OptionalParseResult parseResult =
        parser.parseOptionalAttribute(result, type);
    if (parseResult.hasValue() && succeeded(*parseResult))
      attrs.push_back(parser.builder.getNamedAttr(attrName, result));
    return parseResult;
  }
  OptionalParseResult parseOptionalAttribute(Attribute &result, Type type,
                                             StringRef attrName,
                                             NamedAttrList &attrs) override {
    return parseOptionalAttributeAndAddToList(result, type, attrName, attrs);
  }
  OptionalParseResult parseOptionalAttribute(ArrayAttr &result, Type type,
                                             StringRef attrName,
                                             NamedAttrList &attrs) override {
    return parseOptionalAttributeAndAddToList(result, type, attrName, attrs);
  }
  OptionalParseResult parseOptionalAttribute(StringAttr &result, Type type,
                                             StringRef attrName,
                                             NamedAttrList &attrs) override {
    return parseOptionalAttributeAndAddToList(result, type, attrName, attrs);
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

  /// Returns true if the current token corresponds to a keyword.
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
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Operand Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a single operand.
  ParseResult parseOperand(OperandType &result) override {
    OperationParser::SSAUseInfo useInfo;
    if (parser.parseSSAUse(useInfo))
      return failure();

    result = {useInfo.loc, useInfo.name, useInfo.number};
    return success();
  }

  /// Parse a single operand if present.
  OptionalParseResult parseOptionalOperand(OperandType &result) override {
    if (parser.getToken().is(Token::percent_identifier))
      return parseOperand(result);
    return llvm::None;
  }

  /// Parse zero or more SSA comma-separated operand references with a specified
  /// surrounding delimiter, and an optional required operand count.
  ParseResult parseOperandList(SmallVectorImpl<OperandType> &result,
                               int requiredOperandCount = -1,
                               Delimiter delimiter = Delimiter::None) override {
    return parseOperandOrRegionArgList(result, /*isOperandList=*/true,
                                       requiredOperandCount, delimiter);
  }

  /// Parse zero or more SSA comma-separated operand or region arguments with
  ///  optional surrounding delimiter and required operand count.
  ParseResult
  parseOperandOrRegionArgList(SmallVectorImpl<OperandType> &result,
                              bool isOperandList, int requiredOperandCount = -1,
                              Delimiter delimiter = Delimiter::None) {
    auto startLoc = parser.getToken().getLoc();

    // Handle delimiters.
    switch (delimiter) {
    case Delimiter::None:
      // Don't check for the absence of a delimiter if the number of operands
      // is unknown (and hence the operand list could be empty).
      if (requiredOperandCount == -1)
        break;
      // Token already matches an identifier and so can't be a delimiter.
      if (parser.getToken().is(Token::percent_identifier))
        break;
      // Test against known delimiters.
      if (parser.getToken().is(Token::l_paren) ||
          parser.getToken().is(Token::l_square))
        return emitError(startLoc, "unexpected delimiter");
      return emitError(startLoc, "invalid operand");
    case Delimiter::OptionalParen:
      if (parser.getToken().isNot(Token::l_paren))
        return success();
      LLVM_FALLTHROUGH;
    case Delimiter::Paren:
      if (parser.parseToken(Token::l_paren, "expected '(' in operand list"))
        return failure();
      break;
    case Delimiter::OptionalSquare:
      if (parser.getToken().isNot(Token::l_square))
        return success();
      LLVM_FALLTHROUGH;
    case Delimiter::Square:
      if (parser.parseToken(Token::l_square, "expected '[' in operand list"))
        return failure();
      break;
    }

    // Check for zero operands.
    if (parser.getToken().is(Token::percent_identifier)) {
      do {
        OperandType operandOrArg;
        if (isOperandList ? parseOperand(operandOrArg)
                          : parseRegionArgument(operandOrArg))
          return failure();
        result.push_back(operandOrArg);
      } while (parser.consumeIf(Token::comma));
    }

    // Handle delimiters.   If we reach here, the optional delimiters were
    // present, so we need to parse their closing one.
    switch (delimiter) {
    case Delimiter::None:
      break;
    case Delimiter::OptionalParen:
    case Delimiter::Paren:
      if (parser.parseToken(Token::r_paren, "expected ')' in operand list"))
        return failure();
      break;
    case Delimiter::OptionalSquare:
    case Delimiter::Square:
      if (parser.parseToken(Token::r_square, "expected ']' in operand list"))
        return failure();
      break;
    }

    if (requiredOperandCount != -1 &&
        result.size() != static_cast<size_t>(requiredOperandCount))
      return emitError(startLoc, "expected ")
             << requiredOperandCount << " operands";
    return success();
  }

  /// Parse zero or more trailing SSA comma-separated trailing operand
  /// references with a specified surrounding delimiter, and an optional
  /// required operand count. A leading comma is expected before the operands.
  ParseResult parseTrailingOperandList(SmallVectorImpl<OperandType> &result,
                                       int requiredOperandCount,
                                       Delimiter delimiter) override {
    if (parser.getToken().is(Token::comma)) {
      parseComma();
      return parseOperandList(result, requiredOperandCount, delimiter);
    }
    if (requiredOperandCount != -1)
      return emitError(parser.getToken().getLoc(), "expected ")
             << requiredOperandCount << " operands";
    return success();
  }

  /// Resolve an operand to an SSA value, emitting an error on failure.
  ParseResult resolveOperand(const OperandType &operand, Type type,
                             SmallVectorImpl<Value> &result) override {
    OperationParser::SSAUseInfo operandInfo = {operand.name, operand.number,
                                               operand.location};
    if (auto value = parser.resolveSSAUse(operandInfo, type)) {
      result.push_back(value);
      return success();
    }
    return failure();
  }

  /// Parse an AffineMap of SSA ids.
  ParseResult parseAffineMapOfSSAIds(SmallVectorImpl<OperandType> &operands,
                                     Attribute &mapAttr, StringRef attrName,
                                     NamedAttrList &attrs,
                                     Delimiter delimiter) override {
    SmallVector<OperandType, 2> dimOperands;
    SmallVector<OperandType, 1> symOperands;

    auto parseElement = [&](bool isSymbol) -> ParseResult {
      OperandType operand;
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

  //===--------------------------------------------------------------------===//
  // Region Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a region that takes `arguments` of `argTypes` types.  This
  /// effectively defines the SSA values of `arguments` and assigns their type.
  ParseResult parseRegion(Region &region, ArrayRef<OperandType> arguments,
                          ArrayRef<Type> argTypes,
                          bool enableNameShadowing) override {
    assert(arguments.size() == argTypes.size() &&
           "mismatching number of arguments and types");

    SmallVector<std::pair<OperationParser::SSAUseInfo, Type>, 2>
        regionArguments;
    for (auto pair : llvm::zip(arguments, argTypes)) {
      const OperandType &operand = std::get<0>(pair);
      Type type = std::get<1>(pair);
      OperationParser::SSAUseInfo operandInfo = {operand.name, operand.number,
                                                 operand.location};
      regionArguments.emplace_back(operandInfo, type);
    }

    // Try to parse the region.
    assert((!enableNameShadowing ||
            opDefinition->hasTrait<OpTrait::IsIsolatedFromAbove>()) &&
           "name shadowing is only allowed on isolated regions");
    if (parser.parseRegion(region, regionArguments, enableNameShadowing))
      return failure();
    return success();
  }

  /// Parses a region if present.
  OptionalParseResult parseOptionalRegion(Region &region,
                                          ArrayRef<OperandType> arguments,
                                          ArrayRef<Type> argTypes,
                                          bool enableNameShadowing) override {
    if (parser.getToken().isNot(Token::l_brace))
      return llvm::None;
    return parseRegion(region, arguments, argTypes, enableNameShadowing);
  }

  /// Parses a region if present. If the region is present, a new region is
  /// allocated and placed in `region`. If no region is present, `region`
  /// remains untouched.
  OptionalParseResult
  parseOptionalRegion(std::unique_ptr<Region> &region,
                      ArrayRef<OperandType> arguments, ArrayRef<Type> argTypes,
                      bool enableNameShadowing = false) override {
    if (parser.getToken().isNot(Token::l_brace))
      return llvm::None;
    std::unique_ptr<Region> newRegion = std::make_unique<Region>();
    if (parseRegion(*newRegion, arguments, argTypes, enableNameShadowing))
      return failure();

    region = std::move(newRegion);
    return success();
  }

  /// Parse a region argument. The type of the argument will be resolved later
  /// by a call to `parseRegion`.
  ParseResult parseRegionArgument(OperandType &argument) override {
    return parseOperand(argument);
  }

  /// Parse a region argument if present.
  ParseResult parseOptionalRegionArgument(OperandType &argument) override {
    if (parser.getToken().isNot(Token::percent_identifier))
      return success();
    return parseRegionArgument(argument);
  }

  ParseResult
  parseRegionArgumentList(SmallVectorImpl<OperandType> &result,
                          int requiredOperandCount = -1,
                          Delimiter delimiter = Delimiter::None) override {
    return parseOperandOrRegionArgList(result, /*isOperandList=*/false,
                                       requiredOperandCount, delimiter);
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
    if (parser.getToken().isNot(Token::caret_identifier))
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

  /// Parse a list of assignments of the form
  ///   (%x1 = %y1, %x2 = %y2, ...).
  OptionalParseResult
  parseOptionalAssignmentList(SmallVectorImpl<OperandType> &lhs,
                              SmallVectorImpl<OperandType> &rhs) override {
    if (failed(parseOptionalLParen()))
      return llvm::None;

    auto parseElt = [&]() -> ParseResult {
      OperandType regionArg, operand;
      if (parseRegionArgument(regionArg) || parseEqual() ||
          parseOperand(operand))
        return failure();
      lhs.push_back(regionArg);
      rhs.push_back(operand);
      return success();
    };
    return parser.parseCommaSeparatedListUntil(Token::r_paren, parseElt);
  }

private:
  /// The source location of the operation name.
  SMLoc nameLoc;

  /// Information about the result name specifiers.
  ArrayRef<OperationParser::ResultRecord> resultIDs;

  /// The abstract information of the operation.
  const AbstractOperation *opDefinition;

  /// The main operation parser.
  OperationParser &parser;

  /// A flag that indicates if any errors were emitted during parsing.
  bool emittedError = false;
};
} // end anonymous namespace.

Operation *
OperationParser::parseCustomOperation(ArrayRef<ResultRecord> resultIDs) {
  llvm::SMLoc opLoc = getToken().getLoc();
  StringRef opName = getTokenSpelling();

  auto *opDefinition = AbstractOperation::lookup(opName, getContext());
  if (!opDefinition) {
    if (opName.contains('.')) {
      // This op has a dialect, we try to check if we can register it in the
      // context on the fly.
      StringRef dialectName = opName.split('.').first;
      if (!getContext()->getLoadedDialect(dialectName) &&
          getContext()->getOrLoadDialect(dialectName)) {
        opDefinition = AbstractOperation::lookup(opName, getContext());
      }
    } else {
      // If the operation name has no namespace prefix we treat it as a standard
      // operation and prefix it with "std".
      // TODO: Would it be better to just build a mapping of the registered
      // operations in the standard dialect?
      if (getContext()->getOrLoadDialect("std"))
        opDefinition = AbstractOperation::lookup(Twine("std." + opName).str(),
                                                 getContext());
    }
  }

  if (!opDefinition) {
    emitError(opLoc) << "custom op '" << opName << "' is unknown";
    return nullptr;
  }

  consumeToken();

  // If the custom op parser crashes, produce some indication to help
  // debugging.
  std::string opNameStr = opName.str();
  llvm::PrettyStackTraceFormat fmt("MLIR Parser: custom op parser '%s'",
                                   opNameStr.c_str());

  // Get location information for the operation.
  auto srcLocation = getEncodedSourceLocation(opLoc);

  // Have the op implementation take a crack and parsing this.
  OperationState opState(srcLocation, opDefinition->name);
  CleanupOpStateRegions guard{opState};
  CustomOpAsmParser opAsmParser(opLoc, resultIDs, opDefinition, *this);
  if (opAsmParser.parseOperation(opState))
    return nullptr;

  // If it emitted an error, we failed.
  if (opAsmParser.didEmitError())
    return nullptr;

  // Otherwise, create the operation and try to parse a location for it.
  Operation *op = opBuilder.createOperation(opState);
  if (parseTrailingOperationLocation(op))
    return nullptr;
  return op;
}

ParseResult OperationParser::parseTrailingOperationLocation(Operation *op) {
  // If there is a 'loc' we parse a trailing location.
  if (!consumeIf(Token::kw_loc))
    return success();
  if (parseToken(Token::l_paren, "expected '(' in location"))
    return failure();
  Token tok = getToken();

  // Check to see if we are parsing a location alias.
  LocationAttr directLoc;
  if (tok.is(Token::hash_identifier)) {
    consumeToken();

    StringRef identifier = tok.getSpelling().drop_front();
    if (identifier.contains('.')) {
      return emitError(tok.getLoc())
             << "expected location, but found dialect attribute: '#"
             << identifier << "'";
    }

    // If this alias can be resolved, do it now.
    Attribute attr =
        getState().symbols.attributeAliasDefinitions.lookup(identifier);
    if (attr) {
      if (!(directLoc = attr.dyn_cast<LocationAttr>()))
        return emitError(tok.getLoc())
               << "expected location, but found '" << attr << "'";
    } else {
      // Otherwise, remember this operation and resolve its location later.
      opsWithDeferredLocs.emplace_back(op, tok);
    }

    // Otherwise, we parse the location directly.
  } else if (parseLocationInstance(directLoc)) {
    return failure();
  }

  if (parseToken(Token::r_paren, "expected ')' in location"))
    return failure();

  if (directLoc)
    op->setLoc(directLoc);
  return success();
}

//===----------------------------------------------------------------------===//
// Region Parsing
//===----------------------------------------------------------------------===//

/// Region.
///
///   region ::= '{' region-body
///
ParseResult OperationParser::parseRegion(
    Region &region,
    ArrayRef<std::pair<OperationParser::SSAUseInfo, Type>> entryArguments,
    bool isIsolatedNameScope) {
  // Parse the '{'.
  if (parseToken(Token::l_brace, "expected '{' to begin a region"))
    return failure();

  // Check for an empty region.
  if (entryArguments.empty() && consumeIf(Token::r_brace))
    return success();
  auto currentPt = opBuilder.saveInsertionPoint();

  // Push a new named value scope.
  pushSSANameScope(isIsolatedNameScope);

  // Parse the first block directly to allow for it to be unnamed.
  auto owning_block = std::make_unique<Block>();
  Block *block = owning_block.get();

  // Add arguments to the entry block.
  if (!entryArguments.empty()) {
    for (auto &placeholderArgPair : entryArguments) {
      auto &argInfo = placeholderArgPair.first;
      // Ensure that the argument was not already defined.
      if (auto defLoc = getReferenceLoc(argInfo.name, argInfo.number)) {
        return emitError(argInfo.loc, "region entry argument '" + argInfo.name +
                                          "' is already in use")
                   .attachNote(getEncodedSourceLocation(*defLoc))
               << "previously referenced here";
      }
      if (addDefinition(placeholderArgPair.first,
                        block->addArgument(placeholderArgPair.second))) {
        return failure();
      }
    }

    // If we had named arguments, then don't allow a block name.
    if (getToken().is(Token::caret_identifier))
      return emitError("invalid block name in region with named arguments");
  }

  if (parseBlock(block)) {
    return failure();
  }

  // Verify that no other arguments were parsed.
  if (!entryArguments.empty() &&
      block->getNumArguments() > entryArguments.size()) {
    return emitError("entry block arguments were already defined");
  }

  // Parse the rest of the region.
  region.push_back(owning_block.release());
  if (parseRegionBody(region))
    return failure();

  // Pop the SSA value scope for this region.
  if (popSSANameScope())
    return failure();

  // Reset the original insertion point.
  opBuilder.restoreInsertionPoint(currentPt);
  return success();
}

/// Region.
///
///   region-body ::= block* '}'
///
ParseResult OperationParser::parseRegionBody(Region &region) {
  // Parse the list of blocks.
  while (!consumeIf(Token::r_brace)) {
    Block *newBlock = nullptr;
    if (parseBlock(newBlock))
      return failure();
    region.push_back(newBlock);
  }
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

  block = defineBlockNamed(name, nameLoc, block);

  // Fail if the block was already defined.
  if (!block)
    return emitError(nameLoc, "redefinition of block '") << name << "'";

  // If an argument list is present, parse it.
  if (consumeIf(Token::l_paren)) {
    SmallVector<BlockArgument, 8> bbArgs;
    if (parseOptionalBlockArgList(bbArgs, block) ||
        parseToken(Token::r_paren, "expected ')' to end argument list"))
      return failure();
  }

  if (parseToken(Token::colon, "expected ':' after block name"))
    return failure();

  return parseBlockBody(block);
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
  auto &blockAndLoc = getBlockInfoByName(name);
  if (!blockAndLoc.first) {
    blockAndLoc = {new Block(), loc};
    insertForwardRef(blockAndLoc.first, loc);
  }

  return blockAndLoc.first;
}

/// Define the block with the specified name. Returns the Block* or nullptr in
/// the case of redefinition.
Block *OperationParser::defineBlockNamed(StringRef name, SMLoc loc,
                                         Block *existing) {
  auto &blockAndLoc = getBlockInfoByName(name);
  if (!blockAndLoc.first) {
    // If the caller provided a block, use it.  Otherwise create a new one.
    if (!existing)
      existing = new Block();
    blockAndLoc.first = existing;
    blockAndLoc.second = loc;
    return blockAndLoc.first;
  }

  // Forward declarations are removed once defined, so if we are defining a
  // existing block and it is not a forward declaration, then it is a
  // redeclaration.
  if (!eraseForwardRef(blockAndLoc.first))
    return nullptr;
  return blockAndLoc.first;
}

/// Parse a (possibly empty) list of SSA operands with types as block arguments.
///
///   ssa-id-and-type-list ::= ssa-id-and-type (`,` ssa-id-and-type)*
///
ParseResult OperationParser::parseOptionalBlockArgList(
    SmallVectorImpl<BlockArgument> &results, Block *owner) {
  if (getToken().is(Token::r_brace))
    return success();

  // If the block already has arguments, then we're handling the entry block.
  // Parse and register the names for the arguments, but do not add them.
  bool definingExistingArgs = owner->getNumArguments() != 0;
  unsigned nextArgument = 0;

  return parseCommaSeparatedList([&]() -> ParseResult {
    return parseSSADefOrUseAndType(
        [&](SSAUseInfo useInfo, Type type) -> ParseResult {
          // If this block did not have existing arguments, define a new one.
          if (!definingExistingArgs)
            return addDefinition(useInfo, owner->addArgument(type));

          // Otherwise, ensure that this argument has already been created.
          if (nextArgument >= owner->getNumArguments())
            return emitError("too many arguments specified in argument list");

          // Finally, make sure the existing argument has the correct type.
          auto arg = owner->getArgument(nextArgument++);
          if (arg.getType() != type)
            return emitError("argument and block argument type mismatch");
          return addDefinition(useInfo, arg);
        });
  });
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
  ParseResult parseAttributeAliasDef();

  /// Parse an attribute alias declaration.
  ParseResult parseTypeAliasDef();
};
} // end anonymous namespace

/// Parses an attribute alias declaration.
///
///   attribute-alias-def ::= '#' alias-name `=` attribute-value
///
ParseResult TopLevelOperationParser::parseAttributeAliasDef() {
  assert(getToken().is(Token::hash_identifier));
  StringRef aliasName = getTokenSpelling().drop_front();

  // Check for redefinitions.
  if (getState().symbols.attributeAliasDefinitions.count(aliasName) > 0)
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

  getState().symbols.attributeAliasDefinitions[aliasName] = attr;
  return success();
}

/// Parse a type alias declaration.
///
///   type-alias-def ::= '!' alias-name `=` 'type' type
///
ParseResult TopLevelOperationParser::parseTypeAliasDef() {
  assert(getToken().is(Token::exclamation_identifier));
  StringRef aliasName = getTokenSpelling().drop_front();

  // Check for redefinitions.
  if (getState().symbols.typeAliasDefinitions.count(aliasName) > 0)
    return emitError("redefinition of type alias id '" + aliasName + "'");

  // Make sure this isn't invading the dialect type namespace.
  if (aliasName.contains('.'))
    return emitError("type names with a '.' are reserved for "
                     "dialect-defined names");

  consumeToken(Token::exclamation_identifier);

  // Parse the '=' and 'type'.
  if (parseToken(Token::equal, "expected '=' in type alias definition") ||
      parseToken(Token::kw_type, "expected 'type' in type alias definition"))
    return failure();

  // Parse the type.
  Type aliasedType = parseType();
  if (!aliasedType)
    return failure();

  // Register this alias with the parser state.
  getState().symbols.typeAliasDefinitions.try_emplace(aliasName, aliasedType);
  return success();
}

ParseResult TopLevelOperationParser::parse(Block *topLevelBlock,
                                           Location parserLoc) {
  // Create a top-level operation to contain the parsed state.
  OwningOpRef<Operation *> topLevelOp(ModuleOp::create(parserLoc));
  OperationParser opParser(getState(), topLevelOp.get());
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

      // Verify that the parsed operations are valid.
      if (failed(verify(topLevelOp.get())))
        return failure();

      // Splice the blocks of the parsed operation over to the provided
      // top-level block.
      auto &parsedOps = (*topLevelOp)->getRegion(0).front().getOperations();
      auto &destOps = topLevelBlock->getOperations();
      destOps.splice(destOps.empty() ? destOps.end() : std::prev(destOps.end()),
                     parsedOps, parsedOps.begin(), std::prev(parsedOps.end()));
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
    }
  }
}

//===----------------------------------------------------------------------===//

LogicalResult mlir::parseSourceFile(const llvm::SourceMgr &sourceMgr,
                                    Block *block, MLIRContext *context,
                                    LocationAttr *sourceFileLoc) {
  const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  Location parserLoc = FileLineColLoc::get(
      context, sourceBuf->getBufferIdentifier(), /*line=*/0, /*column=*/0);
  if (sourceFileLoc)
    *sourceFileLoc = parserLoc;

  SymbolState aliasState;
  ParserState state(sourceMgr, context, aliasState);
  return TopLevelOperationParser(state).parse(block, parserLoc);
}

LogicalResult mlir::parseSourceFile(llvm::StringRef filename, Block *block,
                                    MLIRContext *context,
                                    LocationAttr *sourceFileLoc) {
  llvm::SourceMgr sourceMgr;
  return parseSourceFile(filename, sourceMgr, block, context, sourceFileLoc);
}

LogicalResult mlir::parseSourceFile(llvm::StringRef filename,
                                    llvm::SourceMgr &sourceMgr, Block *block,
                                    MLIRContext *context,
                                    LocationAttr *sourceFileLoc) {
  if (sourceMgr.getNumBuffers() != 0) {
    // TODO: Extend to support multiple buffers.
    return emitError(mlir::UnknownLoc::get(context),
                     "only main buffer parsed at the moment");
  }
  auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = file_or_err.getError())
    return emitError(mlir::UnknownLoc::get(context),
                     "could not open input file " + filename);

  // Load the MLIR source file.
  sourceMgr.AddNewSourceBuffer(std::move(*file_or_err), llvm::SMLoc());
  return parseSourceFile(sourceMgr, block, context, sourceFileLoc);
}

LogicalResult mlir::parseSourceString(llvm::StringRef sourceStr, Block *block,
                                      MLIRContext *context,
                                      LocationAttr *sourceFileLoc) {
  auto memBuffer = MemoryBuffer::getMemBuffer(sourceStr);
  if (!memBuffer)
    return failure();

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  return parseSourceFile(sourceMgr, block, context, sourceFileLoc);
}
