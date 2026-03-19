//===- SPIRVTosaOps.cpp - MLIR SPIR-V Tosa operations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tosa operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/InterleavedRange.h"
#include <algorithm>

namespace mlir::spirv {

//===----------------------------------------------------------------------===//
// SPIRV Tosa Custom formatters
//===----------------------------------------------------------------------===//

ParseResult parseSPIRV_I32_1DArmTensor(OpAsmParser &parser,
                                       DenseIntElementsAttr &attr) {
  SmallVector<int32_t, 6> elements;
  auto f = [&]() {
    int32_t value;
    ParseResult r = parser.parseInteger(value);
    elements.push_back(value);
    return r;
  };
  if (parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square, f,
          "parsing values in integer list attribute")) {
    return failure();
  }

  auto i32Type = IntegerType::get(parser.getContext(), 32);
  auto type = TensorArmType::get(
      ArrayRef{static_cast<int64_t>(elements.size())}, i32Type);
  attr = DenseIntElementsAttr::get(type, elements);
  return success();
}

void printSPIRV_I32_1DArmTensor(OpAsmPrinter &printer, Operation *,
                                DenseIntElementsAttr attr) {
  printer << llvm::interleaved_array(
      llvm::map_range(attr.getValues<APInt>(),
                      [](const APInt &a) { return a.getSExtValue(); }));
}

//===----------------------------------------------------------------------===//
// SPIRV Tosa Custom verifiers
//===----------------------------------------------------------------------===//

LogicalResult TosaSelectOp::verify() {
  TensorArmType condType = getConditionType();
  TensorArmType trueValType = getTrueValueType();
  TensorArmType falseValType = getFalseValueType();
  TensorArmType resultType = getResultType();

  if (llvm::any_of(ArrayRef<TensorArmType>{condType, trueValType, falseValType,
                                           resultType},
                   [](TensorArmType type) { return !type.hasRank(); }))
    return success();

  ArrayRef<int64_t> condShape = condType.getShape();
  ArrayRef<int64_t> trueValShape = trueValType.getShape();
  ArrayRef<int64_t> falseValShape = falseValType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();

  if (!llvm::all_equal({condShape.size(), trueValShape.size(),
                        falseValShape.size(), resultShape.size()})) {
    // The AllRanksMatch predicate enforces that all ranks are equal.
    // This is just an extra safe guard for the code coming after that
    // assumes that all ranks are equal.
    return failure();
  }

  for (auto dims :
       llvm::zip_equal(condShape, trueValShape, falseValShape, resultShape)) {
    auto [condDim, trueValDim, falseValDim, resultDim] = dims;

    if (llvm::any_of(
            ArrayRef<int64_t>{condDim, trueValDim, falseValDim, resultDim},
            [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
      continue;
    }

    auto isPairBroadcastable = [](int64_t lhs, int64_t rhs) {
      return lhs == rhs || lhs == 1 || rhs == 1;
    };

    if (!isPairBroadcastable(condDim, trueValDim) ||
        !isPairBroadcastable(condDim, falseValDim) ||
        !isPairBroadcastable(trueValDim, falseValDim)) {
      return emitOpError(
          "failed to verify that the shape of inputs: condition, "
          "true_value, and false_value are compatible for "
          "broadcasting");
    }

    int64_t bradcastedInputDim =
        std::max(condDim, std::max(trueValDim, falseValDim));
    if (bradcastedInputDim != resultDim) {
      return emitOpError(
          "failed to verify that the broadcast shape of inputs: condition, "
          "true_value, and false_value is equal to "
          "the output shape");
    }
  }
  return success();
}

} // namespace mlir::spirv
