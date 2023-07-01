//===- DimLvlMapParser.h - `DimLvlMap` parser -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAPPARSER_H
#define MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAPPARSER_H

#include "DimLvlMap.h"
#include "LvlTypeParser.h"

namespace mlir {
namespace sparse_tensor {
namespace ir_detail {

//===----------------------------------------------------------------------===//
// NOTE(wrengr): The idea here was originally based on the
// "lib/AsmParser/AffineParser.cpp"-static class `AffineParser`.
// Unfortunately, we can't use that class directly since it's file-local.
// Even worse, both `mlir::detail::Parser` and `mlir::detail::ParserState`
// are also file-local classes.  I've been attempting to convert things
// over to using `AsmParser` wherever possible, though it's not clear that
// that'll work...
class DimLvlMapParser final {
public:
  explicit DimLvlMapParser(AsmParser &parser) : parser(parser) {}

  // Parses the input for a sparse tensor dimension-level map
  // and returns the map on success.
  FailureOr<DimLvlMap> parseDimLvlMap();

private:
  // TODO(wrengr): rather than using `OptionalParseResult` and two
  // out-parameters, should we define a type to encapsulate all that?
  OptionalParseResult parseVar(VarKind vk, bool isOptional,
                               CreationPolicy creationPolicy, VarInfo::ID &id,
                               bool &didCreate);
  FailureOr<VarInfo::ID> parseVarUsage(VarKind vk);
  FailureOr<std::pair<Var, bool>> parseVarBinding(VarKind vk, bool isOptional);

  ParseResult parseOptionalSymbolIdList();
  ParseResult parseDimSpec();
  ParseResult parseDimSpecList();
  ParseResult parseLvlSpec();
  ParseResult parseLvlSpecList();

  AsmParser &parser;
  LvlTypeParser lvlTypeParser;
  VarEnv env;
  SmallVector<DimSpec> dimSpecs;
  SmallVector<LvlSpec> lvlSpecs;
};

//===----------------------------------------------------------------------===//

} // namespace ir_detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAPPARSER_H
