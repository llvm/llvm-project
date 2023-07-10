//===- DimLvlMapParser.cpp - `DimLvlMap` parser implementation ------------===//
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

#define FAILURE_IF_FAILED(STMT)                                                \
  if (failed(STMT)) {                                                          \
    return failure();                                                          \
  }

// NOTE: this macro assumes `AsmParser parser` and `SMLoc loc` are in scope.
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
  const auto res = parseVar(vk, /*isOptional=*/false, CreationPolicy::MustNot,
                            varID, didCreate);
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
  }
  return std::make_pair(env.bindUnusedVar(vk), false);
}

FailureOr<Var> DimLvlMapParser::parseLvlVarBinding(bool directAffine) {
  // Nothing to parse, create a new lvl var right away.
  if (directAffine)
    return env.bindUnusedVar(VarKind::Level).cast<LvlVar>();
  // Parse a lvl var, always pulling from the existing pool.
  const auto use = parseVarUsage(VarKind::Level);
  FAILURE_IF_FAILED(use)
  FAILURE_IF_FAILED(parser.parseEqual())
  return env.toVar(*use);
}

//===----------------------------------------------------------------------===//
// `DimLvlMapParser` implementation for `DimLvlMap` per se.
//===----------------------------------------------------------------------===//

FailureOr<DimLvlMap> DimLvlMapParser::parseDimLvlMap() {
  FAILURE_IF_FAILED(parseOptionalIdList(VarKind::Symbol,
                                        OpAsmParser::Delimiter::OptionalSquare))
  FAILURE_IF_FAILED(parseOptionalIdList(VarKind::Level,
                                        OpAsmParser::Delimiter::OptionalBraces))
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

ParseResult
DimLvlMapParser::parseOptionalIdList(VarKind vk,
                                     OpAsmParser::Delimiter delimiter) {
  const auto parseIdBinding = [&]() -> ParseResult {
    return ParseResult(parseVarBinding(vk, /*isOptional=*/false));
  };
  return parser.parseCommaSeparatedList(delimiter, parseIdBinding,
                                        " in id list");
}

//===----------------------------------------------------------------------===//
// `DimLvlMapParser` implementation for `DimSpec`.
//===----------------------------------------------------------------------===//

ParseResult DimLvlMapParser::parseDimSpecList() {
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren,
      [&]() -> ParseResult { return parseDimSpec(); },
      " in dimension-specifier list");
}

ParseResult DimLvlMapParser::parseDimSpec() {
  const auto res = parseVarBinding(VarKind::Dimension, /*isOptional=*/false);
  FAILURE_IF_FAILED(res)
  const DimVar var = res->first.cast<DimVar>();

  // Parse an optional dimension expression.
  AffineExpr affine;
  if (succeeded(parser.parseOptionalEqual())) {
    // Parse the dim affine expr, with only any lvl-vars in scope.
    SmallVector<std::pair<StringRef, AffineExpr>, 4> dimsAndSymbols;
    env.addVars(dimsAndSymbols, VarKind::Level, parser.getContext());
    FAILURE_IF_FAILED(parser.parseAffineExpr(dimsAndSymbols, affine))
  }
  DimExpr expr{affine};

  // Parse an optional slice.
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
  // If no level variable is declared at this point, the following level
  // specification consists of direct affine expressions only, as in:
  //    (d0, d1) -> (d0 : dense, d1 : compressed)
  // Otherwise, we are looking for a leading lvl-var, as in:
  //    {l0, l1} ( d0 = l0, d1 = l1) -> ( l0 = d0 : dense, l1 = d1: compressed)
  const bool directAffine = env.getRanks().getLvlRank() == 0;
  return parser.parseCommaSeparatedList(
      mlir::OpAsmParser::Delimiter::Paren,
      [&]() -> ParseResult { return parseLvlSpec(directAffine); },
      " in level-specifier list");
}

ParseResult DimLvlMapParser::parseLvlSpec(bool directAffine) {
  auto res = parseLvlVarBinding(directAffine);
  FAILURE_IF_FAILED(res);
  LvlVar var = res->cast<LvlVar>();

  // Parse the lvl affine expr, with only the dim-vars in scope.
  AffineExpr affine;
  SmallVector<std::pair<StringRef, AffineExpr>, 4> dimsAndSymbols;
  env.addVars(dimsAndSymbols, VarKind::Dimension, parser.getContext());
  FAILURE_IF_FAILED(parser.parseAffineExpr(dimsAndSymbols, affine))
  LvlExpr expr{affine};

  FAILURE_IF_FAILED(parser.parseColon())

  const auto type = lvlTypeParser.parseLvlType(parser);
  FAILURE_IF_FAILED(type)

  lvlSpecs.emplace_back(var, expr, *type);
  return success();
}

//===----------------------------------------------------------------------===//
