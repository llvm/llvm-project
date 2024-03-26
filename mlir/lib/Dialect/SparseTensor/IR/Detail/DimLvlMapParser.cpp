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

#define FAILURE_IF_FAILED(RES)                                                 \
  if (failed(RES)) {                                                           \
    return failure();                                                          \
  }

/// Helper function for `FAILURE_IF_NULLOPT_OR_FAILED` to avoid duplicating
/// its `RES` parameter.
static inline bool didntSucceed(OptionalParseResult res) {
  return !res.has_value() || failed(*res);
}

#define FAILURE_IF_NULLOPT_OR_FAILED(RES)                                      \
  if (didntSucceed(RES)) {                                                     \
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
                                              Policy creationPolicy,
                                              VarInfo::ID &varID,
                                              bool &didCreate) {
  // Save the current location so that we can have error messages point to
  // the right place.
  const auto loc = parser.getCurrentLocation();
  StringRef name;
  if (failed(parser.parseOptionalKeyword(&name))) {
    ERROR_IF(!isOptional, "expected bare identifier")
    return std::nullopt;
  }

  if (const auto res = env.lookupOrCreate(creationPolicy, name, loc, vk)) {
    varID = res->first;
    didCreate = res->second;
    return success();
  }

  switch (creationPolicy) {
  case Policy::MustNot:
    return parser.emitError(loc, "use of undeclared identifier '" + name + "'");
  case Policy::May:
    llvm_unreachable("got nullopt for Policy::May");
  case Policy::Must:
    return parser.emitError(loc, "redefinition of identifier '" + name + "'");
  }
  llvm_unreachable("unknown Policy");
}

FailureOr<VarInfo::ID> DimLvlMapParser::parseVarUsage(VarKind vk,
                                                      bool requireKnown) {
  VarInfo::ID id;
  bool didCreate;
  const bool isOptional = false;
  const auto creationPolicy = requireKnown ? Policy::MustNot : Policy::May;
  const auto res = parseVar(vk, isOptional, creationPolicy, id, didCreate);
  FAILURE_IF_NULLOPT_OR_FAILED(res)
  assert(requireKnown ? !didCreate : true);
  return id;
}

FailureOr<VarInfo::ID> DimLvlMapParser::parseVarBinding(VarKind vk,
                                                        bool requireKnown) {
  const auto loc = parser.getCurrentLocation();
  VarInfo::ID id;
  bool didCreate;
  const bool isOptional = false;
  const auto creationPolicy = requireKnown ? Policy::MustNot : Policy::Must;
  const auto res = parseVar(vk, isOptional, creationPolicy, id, didCreate);
  FAILURE_IF_NULLOPT_OR_FAILED(res)
  assert(requireKnown ? !didCreate : didCreate);
  bindVar(loc, id);
  return id;
}

FailureOr<std::pair<Var, bool>>
DimLvlMapParser::parseOptionalVarBinding(VarKind vk, bool requireKnown) {
  const auto loc = parser.getCurrentLocation();
  VarInfo::ID id;
  bool didCreate;
  const bool isOptional = true;
  const auto creationPolicy = requireKnown ? Policy::MustNot : Policy::Must;
  const auto res = parseVar(vk, isOptional, creationPolicy, id, didCreate);
  if (res.has_value()) {
    FAILURE_IF_FAILED(*res)
    assert(didCreate);
    return std::make_pair(bindVar(loc, id), true);
  }
  assert(!didCreate);
  return std::make_pair(env.bindUnusedVar(vk), false);
}

Var DimLvlMapParser::bindVar(llvm::SMLoc loc, VarInfo::ID id) {
  MLIRContext *context = parser.getContext();
  const auto var = env.bindVar(id);
  const auto &info = std::as_const(env).access(id);
  const auto name = info.getName();
  const auto num = *info.getNum();
  switch (info.getKind()) {
  case VarKind::Symbol: {
    const auto affine = getAffineSymbolExpr(num, context);
    dimsAndSymbols.emplace_back(name, affine);
    lvlsAndSymbols.emplace_back(name, affine);
    return var;
  }
  case VarKind::Dimension:
    dimsAndSymbols.emplace_back(name, getAffineDimExpr(num, context));
    return var;
  case VarKind::Level:
    lvlsAndSymbols.emplace_back(name, getAffineDimExpr(num, context));
    return var;
  }
  llvm_unreachable("unknown VarKind");
}

//===----------------------------------------------------------------------===//
// `DimLvlMapParser` implementation for `DimLvlMap` per se.
//===----------------------------------------------------------------------===//

FailureOr<DimLvlMap> DimLvlMapParser::parseDimLvlMap() {
  FAILURE_IF_FAILED(parseSymbolBindingList())
  FAILURE_IF_FAILED(parseLvlVarBindingList())
  FAILURE_IF_FAILED(parseDimSpecList())
  FAILURE_IF_FAILED(parser.parseArrow())
  FAILURE_IF_FAILED(parseLvlSpecList())
  InFlightDiagnostic ifd = env.emitErrorIfAnyUnbound(parser);
  if (failed(ifd))
    return ifd;
  return DimLvlMap(env.getRanks().getSymRank(), dimSpecs, lvlSpecs);
}

ParseResult DimLvlMapParser::parseSymbolBindingList() {
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::OptionalSquare,
      [this]() { return ParseResult(parseVarBinding(VarKind::Symbol)); },
      " in symbol binding list");
}

ParseResult DimLvlMapParser::parseLvlVarBindingList() {
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::OptionalBraces,
      [this]() { return ParseResult(parseVarBinding(VarKind::Level)); },
      " in level declaration list");
}

//===----------------------------------------------------------------------===//
// `DimLvlMapParser` implementation for `DimSpec`.
//===----------------------------------------------------------------------===//

ParseResult DimLvlMapParser::parseDimSpecList() {
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren,
      [this]() -> ParseResult { return parseDimSpec(); },
      " in dimension-specifier list");
}

ParseResult DimLvlMapParser::parseDimSpec() {
  // Parse the requisite dim-var binding.
  const auto varID = parseVarBinding(VarKind::Dimension);
  FAILURE_IF_FAILED(varID)
  const DimVar var = env.getVar(*varID).cast<DimVar>();

  // Parse an optional dimension expression.
  AffineExpr affine;
  if (succeeded(parser.parseOptionalEqual())) {
    // Parse the dim affine expr, with only any lvl-vars in scope.
    FAILURE_IF_FAILED(parser.parseAffineExpr(lvlsAndSymbols, affine))
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
  // This method currently only supports two syntaxes:
  //
  // (1) There are no forward-declarations, and no lvl-var bindings:
  //        (d0, d1) -> (d0 : dense, d1 : compressed)
  // Therefore `parseLvlVarBindingList` didn't bind any lvl-vars, and thus
  // `parseLvlSpec` will need to use `VarEnv::bindUnusedVar` to ensure that
  // the level-rank is correct at the end of parsing.
  //
  // (2) There are forward-declarations, and every lvl-spec must have
  // a lvl-var binding:
  //    {l0, l1} (d0 = l0, d1 = l1) -> (l0 = d0 : dense, l1 = d1 : compressed)
  // However, this introduces duplicate information since the order of
  // the lvl-vars in `parseLvlVarBindingList` must agree with their order
  // in the list of lvl-specs.  Therefore, `parseLvlSpec` will not call
  // `VarEnv::bindVar` (since `parseLvlVarBindingList` already did so),
  // and must also validate the consistency between the two lvl-var orders.
  const auto declaredLvlRank = env.getRanks().getLvlRank();
  const bool requireLvlVarBinding = declaredLvlRank != 0;
  // Have `ERROR_IF` point to the start of the list.
  const auto loc = parser.getCurrentLocation();
  const auto res = parser.parseCommaSeparatedList(
      mlir::OpAsmParser::Delimiter::Paren,
      [=]() -> ParseResult { return parseLvlSpec(requireLvlVarBinding); },
      " in level-specifier list");
  FAILURE_IF_FAILED(res)
  const auto specLvlRank = lvlSpecs.size();
  ERROR_IF(requireLvlVarBinding && specLvlRank != declaredLvlRank,
           "Level-rank mismatch between forward-declarations and specifiers. "
           "Declared " +
               Twine(declaredLvlRank) + " level-variables; but got " +
               Twine(specLvlRank) + " level-specifiers.")
  return success();
}

static inline Twine nth(Var::Num n) {
  switch (n) {
  case 1:
    return "1st";
  case 2:
    return "2nd";
  default:
    return Twine(n) + "th";
  }
}

FailureOr<LvlVar>
DimLvlMapParser::parseLvlVarBinding(bool requireLvlVarBinding) {
  // Nothing to parse, just bind an unnamed variable.
  if (!requireLvlVarBinding)
    return env.bindUnusedVar(VarKind::Level).cast<LvlVar>();

  const auto loc = parser.getCurrentLocation();
  // NOTE: Calling `parseVarUsage` here is semantically inappropriate,
  // since the thing we're parsing is supposed to be a variable *binding*
  // rather than a variable *use*.  However, the call to `VarEnv::bindVar`
  // (and its corresponding call to `DimLvlMapParser::recordVarBinding`)
  // already occured in `parseLvlVarBindingList`, and therefore we must
  // use `parseVarUsage` here in order to operationally do the right thing.
  const auto varID = parseVarUsage(VarKind::Level, /*requireKnown=*/true);
  FAILURE_IF_FAILED(varID)
  const auto &info = std::as_const(env).access(*varID);
  const auto var = info.getVar().cast<LvlVar>();
  const auto forwardNum = var.getNum();
  const auto specNum = lvlSpecs.size();
  ERROR_IF(forwardNum != specNum,
           "Level-variable ordering mismatch. The variable '" + info.getName() +
               "' was forward-declared as the " + nth(forwardNum) +
               " level; but is bound by the " + nth(specNum) +
               " specification.")
  FAILURE_IF_FAILED(parser.parseEqual())
  return var;
}

ParseResult DimLvlMapParser::parseLvlSpec(bool requireLvlVarBinding) {
  // Parse the optional lvl-var binding. `requireLvlVarBinding`
  // specifies whether that "optional" is actually Must or MustNot.
  const auto varRes = parseLvlVarBinding(requireLvlVarBinding);
  FAILURE_IF_FAILED(varRes)
  const LvlVar var = *varRes;

  // Parse the lvl affine expr, with only the dim-vars in scope.
  AffineExpr affine;
  FAILURE_IF_FAILED(parser.parseAffineExpr(dimsAndSymbols, affine))
  LvlExpr expr{affine};

  FAILURE_IF_FAILED(parser.parseColon())
  const auto type = lvlTypeParser.parseLvlType(parser);
  FAILURE_IF_FAILED(type)

  lvlSpecs.emplace_back(var, expr, static_cast<LevelType>(*type));
  return success();
}

//===----------------------------------------------------------------------===//
