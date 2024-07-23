//===- QuantOps.cpp - Quantization Type and Ops Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QuantDialectBytecode.h"
#include "TypeDetail.h"

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

#include "mlir/Dialect/Quant/IR/QuantOpsDialect.cpp.inc"


namespace mlir {
namespace quant {

namespace {

// Verify the integrity of per-axis quantization information, if present.
//
// - quantizedType
//   Any quantized type. Any quantized type with no per-axis quantization is
//   ignored.
//
// - containerType
//   Original input or result type of the operation using the provided quantized
//   type. Used to ensure that the quantized type appears within a tensor and
//   that the tensor is compatible with per-axis quantization information.
//
LogicalResult verifyPerAxisQuantization(Operation *op,
                                        QuantizedType quantizedType,
                                        Type containerType) {
  auto quantizedPerAxisType = dyn_cast<UniformQuantizedPerAxisType>(quantizedType);
  if (!quantizedPerAxisType)
    return success();

  auto tensorType = dyn_cast<TensorType>(containerType);
  if (!tensorType)
    return op->emitError("scalar types may not use per-axis quantization");

  if (!tensorType.hasRank())
    return success();

  int64_t quantizedDimension = quantizedPerAxisType.getQuantizedDimension();
  if (quantizedDimension >= tensorType.getRank())
    return op->emitError("quantized dimension must be less than tensor rank");

  int64_t quantizedDimensionSize = tensorType.getDimSize(quantizedDimension);
  if (quantizedDimensionSize != ShapedType::kDynamic &&
      quantizedDimensionSize != (int64_t)quantizedPerAxisType.getScales().size())
    return op->emitError(
        "quantized dimension size does not match number of scales");

  return success();
}

// Common verification logic for 'quant.dcast' and 'quant.qcast' ops.
//
// - quantizedType
//   Quantized type used in the input ('quant.dcast') or result ('quant.qcast'),
//   whether as a primitive type or in a tensor.
//
// - floatType
//   Float type used in the input ('quant.qcast') or result ('quant.dcast'),
//   whether as a primitive type or in a tensor.
//
// - containerType
//   Type of original input or result.
//
LogicalResult verifyQuantizationOp(Operation *op, QuantizedType quantizedType,
                                   FloatType floatType, Type containerType) {
  if (quantizedType.getExpressedType() != floatType)
    return op->emitError(
        "expressed type in quantized type expected to match float type");

  // Veriy integrity of per-axis quantization information, if present.
  return verifyPerAxisQuantization(op, quantizedType, containerType);
}

}  // namespace


//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void QuantDialect::initialize() {
  addTypes<AnyQuantizedType, CalibratedQuantizedType, UniformQuantizedType,
           UniformQuantizedPerAxisType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quant/IR/QuantOps.cpp.inc"
      >();
  detail::addBytecodeInterface(this);
}


//===----------------------------------------------------------------------===//
// DequantizeCastOp
//===----------------------------------------------------------------------===//

LogicalResult DequantizeCastOp::verify() {
  return verifyQuantizationOp(*this, getQuantizedType(), getFloatType(),
                              getInput().getType());
}

FloatType DequantizeCastOp::getFloatType() {
  return cast<FloatType>(getElementTypeOrSelf(getResult().getType()));
}

QuantizedType DequantizeCastOp::getQuantizedType() {
  return cast<QuantizedType>(getElementTypeOrSelf(getInput().getType()));
}


//===----------------------------------------------------------------------===//
// QuantizeCastOp
//===----------------------------------------------------------------------===//

LogicalResult QuantizeCastOp::verify() {
  return verifyQuantizationOp(*this, getQuantizedType(), getFloatType(),
                              getInput().getType());
}

FloatType QuantizeCastOp::getFloatType() {
  return cast<FloatType>(getElementTypeOrSelf(getInput().getType()));
}

QuantizedType QuantizeCastOp::getQuantizedType() {
  return cast<QuantizedType>(getElementTypeOrSelf(getResult().getType()));
}


//===----------------------------------------------------------------------===//
// StorageCastOp
//===----------------------------------------------------------------------===//

IntegerType StorageCastOp::getIntegerType() {
  auto inputScalarType = getElementTypeOrSelf(getInput().getType());
  if (auto integerType = dyn_cast<IntegerType>(inputScalarType))
    return integerType;

  auto resultScalarType = getElementTypeOrSelf(getResult().getType());
  return cast<IntegerType>(resultScalarType);
}

QuantizedType StorageCastOp::getQuantizedType() {
  auto inputScalarType = getElementTypeOrSelf(getInput().getType());
  if (auto quantizedType = dyn_cast<QuantizedType>(inputScalarType))
    return quantizedType;

  auto resultScalarType = getElementTypeOrSelf(getResult().getType());
  return cast<QuantizedType>(resultScalarType);
}

OpFoldResult StorageCastOp::fold(FoldAdaptor adaptor) {
  // Matches x -> [scast -> scast] -> y, replacing the second scast with the
  // value of x if the casts invert each other.
  auto srcScastOp = getInput().getDefiningOp<StorageCastOp>();
  if (!srcScastOp || srcScastOp.getInput().getType() != getType())
    return OpFoldResult();
  return srcScastOp.getInput();
}

LogicalResult StorageCastOp::verify() {
  auto quantizedType = getQuantizedType();
  auto integerType = getIntegerType();
  if (quantizedType.getStorageType() != integerType)
    return emitError(
        "storage type in quantized type expected to match integer type");

  // Verify integrity of per-axis quantization information, if available. While
  // the quantization type may appear in the input or the result, their tensor
  // shapes are guaranteed to be identical at this point.
  return verifyPerAxisQuantization(*this, quantizedType, getInput().getType());
}


} // namespace quant
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Quant/IR/QuantOps.cpp.inc"

