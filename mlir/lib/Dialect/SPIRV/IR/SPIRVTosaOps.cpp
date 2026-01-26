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
#include "llvm/Support/InterleavedRange.h"

namespace mlir::spirv {

//===----------------------------------------------------------------------===//
// TOSA Operator Verifiers.
//===----------------------------------------------------------------------===//

namespace {

LogicalResult verifyConvOp(Operation *op, Type inputETy, Type resultETy,
                           TosaExtAccType accType) {
  if (inputETy.isInteger() && !inputETy.isInteger(8) &&
      !inputETy.isInteger(16)) {
    return op->emitOpError(
        "input element type can only be of width 8 or 16 when integer type");
  }

  if (inputETy.isInteger(8) && !resultETy.isInteger(32)) {
    return op->emitOpError("expect result type to be i32, got ") << resultETy;
  }

  if (inputETy.isInteger(16) && !resultETy.isInteger(64)) {
    return op->emitOpError("expect result type to be i64, got ") << resultETy;
  }

  if (inputETy.isF16() && !resultETy.isF16()) {
    return op->emitOpError("expect result type to be f16, got ") << resultETy;
  }

  if (inputETy.isF32() && !resultETy.isF32()) {
    return op->emitOpError("expect result type to be f32, got ") << resultETy;
  }

  if (inputETy.isInteger(8) && accType != TosaExtAccType::INT32) {
    return op->emitOpError("accumulator type for i8 tensorARM is not i32");
  }

  if (inputETy.isInteger(16) && accType != TosaExtAccType::INT48) {
    return op->emitOpError("accumulator type for i16 tensorARM is not i48");
  }

  if (inputETy.isF16() &&
      !llvm::is_contained({TosaExtAccType::FP16, TosaExtAccType::FP32},
                          accType)) {
    return op->emitOpError(
        "accumulator type for f16 tensorARM is not f16 or f32");
  }

  if (inputETy.isBF16() && accType != TosaExtAccType::FP32) {
    return op->emitOpError("accumulator type for bf16 tensorARM is not f32");
  }

  if (inputETy.isF32() && accType != TosaExtAccType::FP32) {
    return op->emitOpError("accumulator type for f32 tensorARM is not f32");
  }

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// spirv.TosaArgmaxOp
//===----------------------------------------------------------------------===//

LogicalResult TosaArgMaxOp::verify() {
  ShapedType inputTy = getInputType();
  ShapedType resultTy = getResultType();

  if (inputTy.hasRank() && resultTy.hasRank() &&
      resultTy.getRank() !=
          (inputTy.getRank() > 1 ? inputTy.getRank() - 1 : 1)) {
    return emitOpError(
               "result rank must be max of 1 and (input rank - 1), got ")
           << resultTy.getRank();
  }

  const uint32_t axis = getAxis();
  if (inputTy.hasRank() && axis >= inputTy.getRank()) {
    return emitOpError(
               "specified axis is greater than the rank of input, got axis = ")
           << axis << " and input rank = " << inputTy.getRank();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaConv2DOp
//===----------------------------------------------------------------------===//

LogicalResult TosaConv2DOp::verify() {
  Type inputETy = getInputType().getElementType();
  Type resultETy = getResultType().getElementType();
  TosaExtAccType accType = getAccType();
  return verifyConvOp(this->getOperation(), inputETy, resultETy, accType);
}

//===----------------------------------------------------------------------===//
// spirv.TosaConv3DOp
//===----------------------------------------------------------------------===//

LogicalResult TosaConv3DOp::verify() {
  Type inputETy = getInputType().getElementType();
  Type resultETy = getResultType().getElementType();
  TosaExtAccType accType = getAccType();
  return verifyConvOp(this->getOperation(), inputETy, resultETy, accType);
}

//===----------------------------------------------------------------------===//
// SPIRV Tosa DepthwiseConv2D Ops:
//===----------------------------------------------------------------------===//

LogicalResult TosaDepthwiseConv2DOp::verify() {
  Type inputETy = getInputType().getElementType();
  Type resultETy = getResultType().getElementType();
  TosaExtAccType accType = getAccType();
  return verifyConvOp(this->getOperation(), inputETy, resultETy, accType);
}

//===----------------------------------------------------------------------===//
// SPIRV Tosa TransposeConv2D Ops:
//===----------------------------------------------------------------------===//

LogicalResult TosaTransposeConv2DOp::verify() {
  Type inputETy = getInputType().getElementType();
  Type resultETy = getResultType().getElementType();
  TosaExtAccType accType = getAccType();
  return verifyConvOp(this->getOperation(), inputETy, resultETy, accType);
}

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

} // namespace mlir::spirv
