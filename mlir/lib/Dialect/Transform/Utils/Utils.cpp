//===- Utils.cpp - Transform dialect utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Utils/Utils.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mlir::transform;

void transform::printPackedOrDynamicIndexList(OpAsmPrinter &printer,
                                              Operation *op, Value packed,
                                              OperandRange values,
                                              ArrayRef<int64_t> integers) {
  if (packed) {
    assert(values.empty() && integers.empty() && "expected no values/integers");
    printer << packed;
    return;
  }
  printDynamicIndexList(printer, op, values, integers);
}

ParseResult transform::parsePackedOrDynamicIndexList(
    OpAsmParser &parser, std::optional<OpAsmParser::UnresolvedOperand> &packed,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseI64ArrayAttr &integers) {
  OpAsmParser::UnresolvedOperand packedOperand;
  if (parser.parseOptionalOperand(packedOperand).has_value()) {
    packed.emplace(packedOperand);
    integers = parser.getBuilder().getDenseI64ArrayAttr({});
    return success();
  }
  return parseDynamicIndexList(parser, values, integers);
}
