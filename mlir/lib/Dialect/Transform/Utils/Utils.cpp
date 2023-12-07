//===- Utils.cpp - Transform dialect utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Utils/Utils.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mlir::transform;

void mlir::transform::printPackedOrDynamicIndexList(
    OpAsmPrinter &printer, Operation *op, Value packed, Type packedType,
    OperandRange values, TypeRange valueTypes, DenseI64ArrayAttr integers) {
  if (packed) {
    assert(values.empty() && (!integers || integers.empty()) &&
           "expected no values/integers");
    printer << "*(" << packed << " : " << packedType << ")";
    return;
  }
  printDynamicIndexList(printer, op, values, integers, valueTypes);
}

ParseResult mlir::transform::parsePackedOrDynamicIndexList(
    OpAsmParser &parser, std::optional<OpAsmParser::UnresolvedOperand> &packed,
    Type &packedType, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    SmallVectorImpl<Type> &valueTypes, DenseI64ArrayAttr &integers) {
  OpAsmParser::UnresolvedOperand packedOperand;
  if (parser.parseOptionalStar().succeeded()) {
    if (parser.parseLParen().failed() ||
        parser.parseOperand(packedOperand).failed() ||
        parser.parseColonType(packedType).failed() ||
        parser.parseRParen().failed()) {
      return failure();
    }
    packed.emplace(packedOperand);
    integers = parser.getBuilder().getDenseI64ArrayAttr({});
    return success();
  }

  return parseDynamicIndexList(parser, values, integers, &valueTypes);
}
