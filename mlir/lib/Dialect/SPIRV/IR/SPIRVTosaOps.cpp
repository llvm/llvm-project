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

LogicalResult verifyConvolutionOutputDim(int64_t inputSize, int64_t kernelSize,
                                         int64_t outputSize, int64_t padBefore,
                                         int64_t padAfter, int64_t strideSize,
                                         int64_t dilationSize) {
  if (ShapedType::isDynamic(inputSize) || ShapedType::isDynamic(kernelSize))
    return success();

  const int64_t numerator =
      inputSize - 1 + padBefore + padAfter - (kernelSize - 1) * dilationSize;
  if (numerator % strideSize != 0)
    return failure();

  const int64_t calculatedOutput = numerator / strideSize + 1;
  if (!ShapedType::isDynamic(outputSize) && outputSize != calculatedOutput)
    return failure();

  return success();
}

LogicalResult
verifyTransposeConvolutionOutputDim(int64_t inputSize, int64_t kernelSize,
                                    int64_t outputSize, int64_t padBefore,
                                    int64_t padAfter, int64_t strideSize) {
  if (ShapedType::isDynamic(inputSize) || ShapedType::isDynamic(kernelSize))
    return success();

  const int64_t calculatedOutput =
      (inputSize - 1) * strideSize + padBefore + padAfter + kernelSize;
  if (!ShapedType::isDynamic(outputSize) && outputSize != calculatedOutput)
    return failure();

  return success();
}

LogicalResult verifyConv2DOutputShape(Operation *op, DenseIntElementsAttr pad,
                                      DenseIntElementsAttr stride,
                                      DenseIntElementsAttr dilation,
                                      TensorArmType inputType,
                                      TensorArmType weightType,
                                      TensorArmType outputType) {
  constexpr StringLiteral errorMessage =
      "failed to verify that shapes of input, weight, and output must satisfy "
      "[N,IH,IW,*], [*,KH,KW,*], [N,OH,OW,*], with OH = ((IH - 1 + pad_top + "
      "pad_bottom - (KH - 1) * dilation_y) / stride_y) + 1 and OW = ((IW - 1 "
      "+ pad_left + pad_right - (KW - 1) * dilation_x) / stride_x) + 1";
  if (!inputType.hasRank() || !weightType.hasRank() || !outputType.hasRank())
    return success();

  if (failed(verifyConvolutionOutputDim(
          inputType.getDimSize(1), weightType.getDimSize(1),
          outputType.getDimSize(1), getIntValue(pad, 0), getIntValue(pad, 1),
          getIntValue(stride, 0), getIntValue(dilation, 0))))
    return op->emitOpError(errorMessage);

  if (failed(verifyConvolutionOutputDim(
          inputType.getDimSize(2), weightType.getDimSize(2),
          outputType.getDimSize(2), getIntValue(pad, 2), getIntValue(pad, 3),
          getIntValue(stride, 1), getIntValue(dilation, 1))))
    return op->emitOpError(errorMessage);

  return success();
}

LogicalResult verifyConv3DOutputShape(Operation *op, DenseIntElementsAttr pad,
                                      DenseIntElementsAttr stride,
                                      DenseIntElementsAttr dilation,
                                      TensorArmType inputType,
                                      TensorArmType weightType,
                                      TensorArmType outputType) {
  constexpr StringLiteral errorMessage =
      "failed to verify that shapes of input, weight, and output must satisfy "
      "[N,ID,IH,IW,*], [*,KD,KH,KW,*], [N,OD,OH,OW,*], with OD = ((ID - 1 + "
      "pad_front + pad_back - (KD - 1) * dilation_d) / stride_d) + 1, OH = "
      "((IH - 1 + pad_top + pad_bottom - (KH - 1) * dilation_y) / stride_y) "
      "+ 1 and OW = ((IW - 1 + pad_left + pad_right - (KW - 1) * dilation_x) "
      "/ stride_x) + 1";
  if (!inputType.hasRank() || !weightType.hasRank() || !outputType.hasRank())
    return success();

  if (failed(verifyConvolutionOutputDim(
          inputType.getDimSize(1), weightType.getDimSize(1),
          outputType.getDimSize(1), getIntValue(pad, 0), getIntValue(pad, 1),
          getIntValue(stride, 0), getIntValue(dilation, 0))))
    return op->emitOpError(errorMessage);

  if (failed(verifyConvolutionOutputDim(
          inputType.getDimSize(2), weightType.getDimSize(2),
          outputType.getDimSize(2), getIntValue(pad, 2), getIntValue(pad, 3),
          getIntValue(stride, 1), getIntValue(dilation, 1))))
    return op->emitOpError(errorMessage);

  if (failed(verifyConvolutionOutputDim(
          inputType.getDimSize(3), weightType.getDimSize(3),
          outputType.getDimSize(3), getIntValue(pad, 4), getIntValue(pad, 5),
          getIntValue(stride, 2), getIntValue(dilation, 2))))
    return op->emitOpError(errorMessage);

  return success();
}

LogicalResult verifyDepthwiseConv2DOutputShape(
    Operation *op, DenseIntElementsAttr pad, DenseIntElementsAttr stride,
    DenseIntElementsAttr dilation, TensorArmType inputType,
    TensorArmType weightType, TensorArmType outputType) {
  constexpr StringLiteral errorMessage =
      "failed to verify that shapes of input, weight, and output must satisfy "
      "[N,IH,IW,*], [KH,KW,*,*], [N,OH,OW,*], with OH = ((IH - 1 + pad_top + "
      "pad_bottom - (KH - 1) * dilation_y) / stride_y) + 1 and OW = ((IW - 1 "
      "+ pad_left + pad_right - (KW - 1) * dilation_x) / stride_x) + 1";
  if (!inputType.hasRank() || !weightType.hasRank() || !outputType.hasRank())
    return success();

  if (failed(verifyConvolutionOutputDim(
          inputType.getDimSize(1), weightType.getDimSize(0),
          outputType.getDimSize(1), getIntValue(pad, 0), getIntValue(pad, 1),
          getIntValue(stride, 0), getIntValue(dilation, 0))))
    return op->emitOpError(errorMessage);

  if (failed(verifyConvolutionOutputDim(
          inputType.getDimSize(2), weightType.getDimSize(1),
          outputType.getDimSize(2), getIntValue(pad, 2), getIntValue(pad, 3),
          getIntValue(stride, 1), getIntValue(dilation, 1))))
    return op->emitOpError(errorMessage);

  return success();
}

LogicalResult verifyTransposeConv2DOutputShape(Operation *op,
                                               DenseIntElementsAttr outPad,
                                               DenseIntElementsAttr stride,
                                               TensorArmType inputType,
                                               TensorArmType weightType,
                                               TensorArmType outputType) {
  constexpr StringLiteral errorMessage =
      "failed to verify that shapes of input, weight, and output must satisfy "
      "[N,IH,IW,*], [*,KH,KW,*], [N,OH,OW,*], with OH = (IH - 1) * stride_y + "
      "out_pad_top + out_pad_bottom + KH and OW = (IW - 1) * stride_x + "
      "out_pad_left + out_pad_right + KW";
  if (!inputType.hasRank() || !weightType.hasRank() || !outputType.hasRank())
    return success();

  const int64_t kernelHeight = weightType.getDimSize(1);
  if (ShapedType::isStatic(kernelHeight) &&
      (getIntValue(outPad, 0) <= -kernelHeight ||
       getIntValue(outPad, 1) <= -kernelHeight))
    return op->emitOpError("expected out_pad_top and out_pad_bottom to be > "
                           "-KH");

  const int64_t kernelWidth = weightType.getDimSize(2);
  if (ShapedType::isStatic(kernelWidth) &&
      (getIntValue(outPad, 2) <= -kernelWidth ||
       getIntValue(outPad, 3) <= -kernelWidth))
    return op->emitOpError("expected out_pad_left and out_pad_right to be > "
                           "-KW");

  if (failed(verifyTransposeConvolutionOutputDim(
          inputType.getDimSize(1), kernelHeight, outputType.getDimSize(1),
          getIntValue(outPad, 0), getIntValue(outPad, 1),
          getIntValue(stride, 0))))
    return op->emitOpError(errorMessage);

  if (failed(verifyTransposeConvolutionOutputDim(
          inputType.getDimSize(2), kernelWidth, outputType.getDimSize(2),
          getIntValue(outPad, 2), getIntValue(outPad, 3),
          getIntValue(stride, 1))))
    return op->emitOpError(errorMessage);

  return success();
}

} // namespace

LogicalResult TosaAvgPool2DOp::verify() {
  return verifyPool2DOp(getOperation(), getKernel(), getStride(), getPad(),
                        getInputType(), getResultType());
}

LogicalResult TosaConv2DOp::verify() {
  return verifyConv2DOutputShape(getOperation(), getPad(), getStride(),
                                 getDilation(), getInputType(), getWeightType(),
                                 getResultType());
}

LogicalResult TosaConv3DOp::verify() {
  return verifyConv3DOutputShape(getOperation(), getPad(), getStride(),
                                 getDilation(), getInputType(), getWeightType(),
                                 getResultType());
}

LogicalResult TosaDepthwiseConv2DOp::verify() {
  return verifyDepthwiseConv2DOutputShape(getOperation(), getPad(), getStride(),
                                          getDilation(), getInputType(),
                                          getWeightType(), getResultType());
}

LogicalResult TosaMaxPool2DOp::verify() {
  return verifyPool2DOp(getOperation(), getKernel(), getStride(), getPad(),
                        getInputType(), getResultType());
}

LogicalResult TosaTransposeConv2DOp::verify() {
  return verifyTransposeConv2DOutputShape(getOperation(), getOutPad(),
                                          getStride(), getInputType(),
                                          getWeightType(), getResultType());
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
