//===- DimLvlMapParser.cpp - `DimLvlMap` parser implementation ------------===//
// These two lookup methods are probably small enough to benefit from
// being defined inline/in-class, expecially since doing so may allow the
// compiler to optimize the `std::optional` away.  But we put the defns
// here until benchmarks prove the benefit of doing otherwise.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DimLvlMapParser.h"

using namespace mlir;
using namespace mlir::sparse_tensor;
using namespace mlir::sparse_tensor::ir_detail;

//===----------------------------------------------------------------------===//
// TODO(wrengr): rephrase these to do the trick for gobbling up any trailing
// semicolon
//
// NOTE: There's no way for `FAILURE_IF_FAILED` to simultaneously support
// both `OptionalParseResult` and `InFlightDiagnostic` return types.
// We can get the compiler to accept the code if we returned "`{}`",
// however for `OptionalParseResult` that would become the nullopt result,
// whereas for `InFlightDiagnostic` it would become a result that can
// be implicitly converted to success.  By using "`failure()`" we ensure
// that `OptionalParseResult` behaves as intended, however that means the
// macro cannot be used for `InFlightDiagnostic` since there's no implicit
// conversion.
#define FAILURE_IF_FAILED(STMT)                                                \
  if (failed(STMT)) {                                                          \
    return failure();                                                          \
  }

// Although `ERROR_IF` is phrased to return `InFlightDiagnostic`, that type
// can be implicitly converted to all four of `LogicalResult, `FailureOr`,
// `ParseResult`, and `OptionalParseResult`.  (However, beware that the
// conversion to `OptionalParseResult` doesn't properly delegate to
// `InFlightDiagnostic::operator ParseResult`.)
//
// NOTE: this macro assumes `AsmParser parser` and `SMLoc loc` are in scope.
// NOTE_TO_SELF(wrengr): The LOC used to always be `parser.getNameLoc()`
#define ERROR_IF(COND, MSG)                                                    \
  if (COND) {                                                                  \
    return parser.emitError(loc, MSG);                                         \
  }

//===----------------------------------------------------------------------===//
// `DimLvlMapParser` implementation for variable parsing.
//===----------------------------------------------------------------------===//

// Our variation on `AffineParser::{parseBareIdExpr,parseIdentifierDefinition}`
OptionalParseResult DimLvlMapParser::parseVar(VarKind vk, bool isOptional,
                                              CreationPolicy creationPolicy,
                                              VarInfo::ID &varID,
                                              bool &didCreate) {
  // Save the current location so that we can have error messages point to
  // the right place.  Note that `Parser::emitWrongTokenError` starts off
  // with the same location as `AsmParserImpl::getCurrentLocation` returns;
  // however, `Parser` will then do some various munging with the location
  // before actually using it, so `AsmParser::emitError` can't quite be used
  // as a drop-in replacement for `Parser::emitWrongTokenError`.
  const auto loc = parser.getCurrentLocation();

  // Several things to note.
  // (1) the `Parser::isCurrentTokenAKeyword` method checks the exact
  //     same conditions as the `AffineParser.cpp`-static free-function
  //     `isIdentifier` which is used by `AffineParser::parseBareIdExpr`.
  // (2) the `{Parser,AsmParserImpl}::parseOptionalKeyword(StringRef*)`
  //     methods do the same song and dance about using
  //     `isCurrentTokenAKeyword`, `getTokenSpelling`, et `consumeToken` as we
  //     would want to do if we could use the `Parser` class directly.  It
  //     doesn't provide the nice error handling we want, but we can work around
  //     that.
  StringRef name;
  if (failed(parser.parseOptionalKeyword(&name))) {
    // If not actually optional, then `emitError`.
    ERROR_IF(!isOptional, "expected bare identifier")
    // If is actually optional, then return the null `OptionalParseResult`.
    return std::nullopt;
  }

  // I don't know if we need to worry about the possibility of the caller
  // recovering from error and then reusing the `DimLvlMapParser` for subsequent
  // `parseVar`, but I'm erring on the side of caution by distinguishing
  // all three possible creation policies.
  if (const auto res = env.lookupOrCreate(creationPolicy, name, loc, vk)) {
    varID = res->first;
    didCreate = res->second;
    return success();
  }
  // TODO(wrengr): these error messages make sense for our intended usage,
  // but not in general; but it's unclear how best to factor that part out.
  switch (creationPolicy) {
  case CreationPolicy::MustNot:
    return parser.emitError(loc, "use of undeclared identifier '" + name + "'");
  case CreationPolicy::May:
    llvm_unreachable("got nullopt for CreationPolicy::May");
  case CreationPolicy::Must:
    return parser.emitError(loc, "redefinition of identifier '" + name + "'");
  }
  llvm_unreachable("unknown CreationPolicy");
}

FailureOr<VarInfo::ID> DimLvlMapParser::parseVarUsage(VarKind vk) {
  VarInfo::ID varID;
  bool didCreate;
  // We use the policy `May` because we want to allow parsing free/unbound
  // variables.  If we wanted to distinguish between parsing free-var uses
  // vs bound-var uses, then the latter should use `MustNot`.
  const auto res =
      parseVar(vk, /*isOptional=*/false, CreationPolicy::May, varID, didCreate);
  if (!res.has_value() || failed(*res))
    return failure();
  return varID;
}

FailureOr<std::pair<Var, bool>>
DimLvlMapParser::parseVarBinding(VarKind vk, bool isOptional) {
  VarInfo::ID id;
  bool didCreate;
  const auto res =
      parseVar(vk, isOptional, CreationPolicy::Must, id, didCreate);
  if (res.has_value()) {
    FAILURE_IF_FAILED(*res)
    return std::make_pair(env.bindVar(id), true);
  } else {
    return std::make_pair(env.bindUnusedVar(vk), false);
  }
}

//===----------------------------------------------------------------------===//
// `DimLvlMapParser` implementation for `DimLvlMap` per se.
//===----------------------------------------------------------------------===//

FailureOr<DimLvlMap> DimLvlMapParser::parseDimLvlMap() {
  FAILURE_IF_FAILED(parseOptionalSymbolIdList())
  FAILURE_IF_FAILED(parseDimSpecList())
  FAILURE_IF_FAILED(parser.parseArrow())
  FAILURE_IF_FAILED(parseLvlSpecList())
  // TODO(wrengr): Try to improve the error messages from
  // `VarEnv::emitErrorIfAnyUnbound`.
  InFlightDiagnostic ifd = env.emitErrorIfAnyUnbound(parser);
  if (failed(ifd))
    return ifd;
  return DimLvlMap(env.getRanks().getSymRank(), dimSpecs, lvlSpecs);
}

using Delimiter = mlir::OpAsmParser::Delimiter;

ParseResult DimLvlMapParser::parseOptionalSymbolIdList() {
  const auto parseSymVarBinding = [&]() -> ParseResult {
    return ParseResult(parseVarBinding(VarKind::Symbol, /*isOptional=*/false));
  };
  // If I've correctly unpacked how exactly `Parser::parseCommaSeparatedList`
  // handles the "optional" delimiters vs the non-optional ones, then
  // the following call to `AsmParser::parseCommaSeparatedList` should
  // be equivalent to the whole `AffineParse::parseOptionalSymbolIdList`
  // method (which uses `Parser` methods to handle the optionality instead).
  return parser.parseCommaSeparatedList(Delimiter::OptionalSquare,
                                        parseSymVarBinding, " in symbol list");
}

//===----------------------------------------------------------------------===//
// `DimLvlMapParser` implementation for `DimSpec`.
//===----------------------------------------------------------------------===//

ParseResult DimLvlMapParser::parseDimSpecList() {
  return parser.parseCommaSeparatedList(
      Delimiter::Paren, [&]() -> ParseResult { return parseDimSpec(); },
      " in dimension-specifier list");
}

ParseResult DimLvlMapParser::parseDimSpec() {
  const auto res = parseVarBinding(VarKind::Dimension, /*isOptional=*/false);
  FAILURE_IF_FAILED(res)
  const DimVar var = res->first.cast<DimVar>();

  DimExpr expr{AffineExpr()};
  if (succeeded(parser.parseOptionalEqual())) {
    // FIXME(wrengr): I don't think there's any way to implement this
    // without replicating the bulk of `AffineParser::parseAffineExpr`
    // TODO(wrengr): Also, need to make sure the parser uses
    // `parseVarUsage(VarKind::Level)` so that every `AffineDimExpr`
    // necessarily corresponds to a `LvlVar` (never a `DimVar`).
    //
    // FIXME: proof of concept, parse trivial level vars (viz d0 = l0).
    auto use = parseVarUsage(VarKind::Level);
    FAILURE_IF_FAILED(use)
    AffineExpr a = getAffineDimExpr(var.getNum(), parser.getContext());
    DimExpr dexpr{a};
    expr = dexpr;
  }

  SparseTensorDimSliceAttr slice;
  if (succeeded(parser.parseOptionalColon())) {
    const auto loc = parser.getCurrentLocation();
    Attribute attr;
    FAILURE_IF_FAILED(parser.parseAttribute(attr))
    slice = llvm::dyn_cast<SparseTensorDimSliceAttr>(attr);
    ERROR_IF(!slice, "expected SparseTensorDimSliceAttr")
  }

  dimSpecs.emplace_back(var, expr, slice);
  return success();
}

//===----------------------------------------------------------------------===//
// `DimLvlMapParser` implementation for `LvlSpec`.
//===----------------------------------------------------------------------===//

ParseResult DimLvlMapParser::parseLvlSpecList() {
  return parser.parseCommaSeparatedList(
      Delimiter::Paren, [&]() -> ParseResult { return parseLvlSpec(); },
      " in level-specifier list");
}

ParseResult DimLvlMapParser::parseLvlSpec() {
  // FIXME(wrengr): This implementation isn't actually going to work as-is,
  // due to grammar ambiguity.  That is, assuming the current token is indeed
  // a variable, we don't yet know whether that variable is supposed to
  // be a binding vs being a usage that's part of the following AffineExpr.
  // We can only disambiguate that by peeking at the next token to see whether
  // it's the equals symbol or not.
  //
  // FIXME: proof of concept, assume it is new (viz. l0 = d0).
  const auto res = parseVarBinding(VarKind::Level, /*isOptional=*/true);
  FAILURE_IF_FAILED(res)
  if (res->second) {
    FAILURE_IF_FAILED(parser.parseEqual())
  }
  const LvlVar var = res->first.cast<LvlVar>();

  // FIXME(wrengr): I don't think there's any way to implement this
  // without replicating the bulk of `AffineParser::parseAffineExpr`
  //
  // TODO(wrengr): Also, need to make sure the parser uses
  // `parseVarUsage(VarKind::Dimension)` so that every `AffineDimExpr`
  // necessarily corresponds to a `DimVar` (never a `LvlVar`).
  //
  // FIXME: proof of concept, parse trivial dim vars (viz l0 = d0).
  auto use = parseVarUsage(VarKind::Dimension);
  FAILURE_IF_FAILED(use)
  AffineExpr a =
      getAffineDimExpr(env.toVar(*use).getNum(), parser.getContext());
  LvlExpr expr{a};

  FAILURE_IF_FAILED(parser.parseColon())

  const auto type = lvlTypeParser.parseLvlType(parser);
  FAILURE_IF_FAILED(type)

  lvlSpecs.emplace_back(var, expr, *type);
  return success();
}

//===----------------------------------------------------------------------===//
