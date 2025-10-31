//===- SPIRVOpDefinition.cpp - MLIR SPIR-V Op Definition Implementation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the TableGen'erated SPIR-V op implementation in the SPIR-V dialect.
// These are placed in a separate file to reduce the total amount of code in
// SPIRVOps.cpp and make that file faster to recompile.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "SPIRVParsingUtils.h"

#include "mlir/IR/TypeUtilities.h"

namespace mlir::spirv {
/// Returns true if the given op is a function-like op or nested in a
/// function-like op without a module-like op in the middle.
static bool isNestedInFunctionOpInterface(Operation *op) {
  if (!op)
    return false;
  if (op->hasTrait<OpTrait::SymbolTable>())
    return false;
  if (isa<FunctionOpInterface>(op))
    return true;
  return isNestedInFunctionOpInterface(op->getParentOp());
}

/// Returns true if the given op is a GraphARM op or nested in a
/// GraphARM op without a module-like op in the middle.
static bool isNestedInGraphARMOpInterface(Operation *op) {
  if (!op)
    return false;
  if (op->hasTrait<OpTrait::SymbolTable>())
    return false;
  if (isa<spirv::GraphARMOp>(op))
    return true;
  return isNestedInGraphARMOpInterface(op->getParentOp());
}

/// Returns true if the given op is an module-like op that maintains a symbol
/// table.
static bool isDirectInModuleLikeOp(Operation *op) {
  return op && op->hasTrait<OpTrait::SymbolTable>();
}

/// Result of a logical op must be a scalar or vector of boolean type.
static Type getUnaryOpResultType(Type operandType) {
  Builder builder(operandType.getContext());
  Type resultType = builder.getIntegerType(1);
  if (auto vecType = llvm::dyn_cast<VectorType>(operandType))
    return VectorType::get(vecType.getNumElements(), resultType);
  return resultType;
}

static ParseResult parseImageOperands(OpAsmParser &parser,
                                      spirv::ImageOperandsAttr &attr) {
  // Expect image operands
  if (parser.parseOptionalLSquare())
    return success();

  spirv::ImageOperands imageOperands;
  if (parseEnumStrAttr(imageOperands, parser))
    return failure();

  attr = spirv::ImageOperandsAttr::get(parser.getContext(), imageOperands);

  return parser.parseRSquare();
}

static void printImageOperands(OpAsmPrinter &printer, Operation *imageOp,
                               spirv::ImageOperandsAttr attr) {
  if (attr) {
    auto strImageOperands = stringifyImageOperands(attr.getValue());
    printer << "[\"" << strImageOperands << "\"]";
  }
}

} // namespace mlir::spirv

// TablenGen'erated operation definitions.
#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.cpp.inc"
