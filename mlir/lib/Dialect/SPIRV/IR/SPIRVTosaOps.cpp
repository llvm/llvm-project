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

namespace {

int64_t getIntValue(DenseIntElementsAttr attr, size_t idx) {
  return attr.getValues<APInt>()[idx].getSExtValue();
}

LogicalResult verifyPool2DOutputDim(Operation *op, int64_t inputSize,
                                    int64_t outputSize, int64_t kernelSize,
                                    int64_t strideSize, int64_t padBefore,
                                    int64_t padAfter, StringRef dimName,
                                    StringRef dimAxis, StringRef padBeforeName,
                                    StringRef padAfterName) {
  if (ShapedType::isDynamic(inputSize))
    return success();

  const int64_t numerator = inputSize + padBefore + padAfter - kernelSize;
  if (numerator % strideSize != 0)
    return op->emitOpError("expected input_")
           << dimName << " + pad_" << padBeforeName << " + pad_" << padAfterName
           << " - kernel_" << dimAxis << " to be wholly divisible by stride_"
           << dimAxis << ", got (" << inputSize << " + " << padBefore << " + "
           << padAfter << " - " << kernelSize << ") / " << strideSize;

  const int64_t calculatedOutput = numerator / strideSize + 1;
  if (!ShapedType::isDynamic(outputSize) && outputSize != calculatedOutput)
    return op->emitOpError("failed to verify that shapes of input and output "
                           "must satisfy [N,IH,IW,C] and [N,OH,OW,C], with "
                           "OH = ((IH + pad_top + pad_bottom - kernel_y) / "
                           "stride_y) + 1 and OW = ((IW + pad_left + "
                           "pad_right - kernel_x) / stride_x) + 1");

  return success();
}

LogicalResult verifyPool2DOp(Operation *op, DenseIntElementsAttr kernel,
                             DenseIntElementsAttr stride,
                             DenseIntElementsAttr pad, TensorArmType inputType,
                             TensorArmType outputType) {

  if (!inputType.hasRank() || !outputType.hasRank())
    return success();

  if (failed(verifyPool2DOutputDim(
          op, inputType.getDimSize(1), outputType.getDimSize(1),
          getIntValue(kernel, 0), getIntValue(stride, 0), getIntValue(pad, 0),
          getIntValue(pad, 1), "height", "y", "top", "bottom")))
    return failure();

  if (failed(verifyPool2DOutputDim(
          op, inputType.getDimSize(2), outputType.getDimSize(2),
          getIntValue(kernel, 1), getIntValue(stride, 1), getIntValue(pad, 2),
          getIntValue(pad, 3), "width", "x", "left", "right")))
    return failure();

  return success();
}

} // namespace

LogicalResult TosaAvgPool2DOp::verify() {
  return verifyPool2DOp(getOperation(), getKernel(), getStride(), getPad(),
                        getInputType(), getResultType());
}

LogicalResult TosaMaxPool2DOp::verify() {
  return verifyPool2DOp(getOperation(), getKernel(), getStride(), getPad(),
                        getInputType(), getResultType());
}

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
