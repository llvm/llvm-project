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

///
/// Parses the Sparse Tensor Encoding Attribute (STEA).
///
/// General syntax is as follows,
///
///   [s0, ...]     // optional forward decl sym-vars
///   {l0, ...}     // optional forward decl lvl-vars
///   (
///     d0 = ...,   // dim-var = dim-exp
///     ...
///   ) -> (
///     l0 = ...,   // lvl-var = lvl-exp
///     ...
///   )
///
/// with simplifications when variables are implicit.
///
class DimLvlMapParser final {
public:
  explicit DimLvlMapParser(AsmParser &parser) : parser(parser) {}

  // Parses the input for a sparse tensor dimension-level map
  // and returns the map on success.
  FailureOr<DimLvlMap> parseDimLvlMap();

private:
  OptionalParseResult parseVar(VarKind vk, bool isOptional,
                               CreationPolicy creationPolicy, VarInfo::ID &id,
                               bool &didCreate);
  FailureOr<VarInfo::ID> parseVarUsage(VarKind vk);
  FailureOr<std::pair<Var, bool>> parseVarBinding(VarKind vk, bool isOptional);
  FailureOr<Var> parseLvlVarBinding(bool directAffine);

  ParseResult parseOptionalIdList(VarKind vk, OpAsmParser::Delimiter delimiter);
  ParseResult parseDimSpec();
  ParseResult parseDimSpecList();
  ParseResult parseLvlSpec(bool directAffine);
  ParseResult parseLvlSpecList();

  AsmParser &parser;
  LvlTypeParser lvlTypeParser;
  VarEnv env;
  SmallVector<DimSpec> dimSpecs;
  SmallVector<LvlSpec> lvlSpecs;
};

} // namespace ir_detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAPPARSER_H
