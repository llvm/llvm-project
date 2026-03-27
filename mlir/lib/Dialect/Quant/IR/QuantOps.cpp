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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/Dialect/Quant/IR/QuantOpsDialect.cpp.inc"

namespace mlir {
namespace quant {

namespace {

// Verify the integrity of per-axis quantization information, if present.
//
// - uniformQuantizedPerAxisType
//   A quantized type with per-axis quantization.
//
// - containerType
//   Original input or result type of the operation using the provided quantized
//   type. Used to ensure that the quantized type appears within a tensor and
//   that the tensor is compatible with per-axis quantization information.
//
LogicalResult verifyPerAxisQuantization(
    Operation *op, UniformQuantizedPerAxisType uniformQuantizedPerAxisType,
    Type containerType) {
  auto tensorType = dyn_cast<TensorType>(containerType);
  if (!tensorType)
    return op->emitError("scalar types may not use per-axis quantization");

  if (!tensorType.hasRank())
    return success();

  int32_t quantizedDimension =
      uniformQuantizedPerAxisType.getQuantizedDimension();
  if ((int64_t)quantizedDimension >= tensorType.getRank())
    return op->emitError("quantized dimension must be less than tensor rank");

  int64_t quantizedDimensionSize = tensorType.getDimSize(quantizedDimension);
  if (quantizedDimensionSize != ShapedType::kDynamic &&
      quantizedDimensionSize !=
          (int64_t)uniformQuantizedPerAxisType.getScales().size())
    return op->emitError(
        "quantized dimension size does not match number of scales");

  return success();
}

// Verifies that the sub-channel quantization parameters are consistent with
// the given container type. The function checks the following:
//
// - The container type must be a ranked tensor type.
// - Each quantized dimension must be less than the rank of the tensor.
// - The size of each dimension at the quantized dimension must be divisible
//    by the corresponding block size.
// - The scale dimension size at each axis index should match the tensor
//    dimension at the index divided by the corresponding block size.
//
// The `uniformQuantizedSubChannelType` argument provides the sub-channel
// quantization parameters, and the `containerType` argument specifies the
// type of the container holding the quantized data.
//
LogicalResult verifySubChannelQuantization(
    Operation *op,
    UniformQuantizedSubChannelType uniformQuantizedSubChannelType,
    Type containerType) {
  auto tensorType = dyn_cast<TensorType>(containerType);
  if (!tensorType)
    return op->emitError("scalar types may not use sub-channel quantization");

  if (!tensorType.hasRank())
    return op->emitError(
        "tensor containing the sub-channel quantized type must be ranked");

  const SmallVector<std::pair<int32_t, int64_t>> &blockSizeInfo =
      uniformQuantizedSubChannelType.getBlockSizeInfo();
  auto shape = tensorType.getShape();

  // The dimension size of scale for an axis which is not specified as quantized
  // dimension should be 1.
  SmallVector<int64_t> expectedScaleShape(tensorType.getShape().size(), 1);
  for (auto [quantizedDimension, blockSize] : blockSizeInfo) {
    if (quantizedDimension >= tensorType.getRank())
      return op->emitError()
             << "quantized dimension " << quantizedDimension
             << " must be less than tensor rank " << tensorType.getRank();
    if (!tensorType.isDynamicDim(quantizedDimension) &&
        tensorType.getDimSize(quantizedDimension) % blockSize != 0)
      return op->emitError()
             << "tensor dimension size "
             << tensorType.getDimSize(quantizedDimension) << " at axis "
             << quantizedDimension
             << " must be divisible by the corresponding block size "
             << blockSize;
    if (tensorType.isDynamicDim(quantizedDimension))
      expectedScaleShape[quantizedDimension] = ShapedType::kDynamic;
    else
      expectedScaleShape[quantizedDimension] =
          tensorType.getDimSize(quantizedDimension) / blockSize;
  }

  // Block sizes must be greater than 0 and divide the corresponding dimension
  // size. While a block size b must be less than or equal to the corresponding
  // dimension size d, this constraint is implicitly enforced by requiring that
  // d % b == 0 when d != 0.
  //
  // However, a problem arises when d = 0.  The divisibility constraint allows b
  // to be any value, potentially violating the requirement that b <= d.
  // Furthermore, if b is unspecified (implicitly equal to d), it violates the
  // constraint that b > 0.
  //
  // Therefore, we explicitly disallow the case where d = 0 to maintain
  // consistency and avoid these issues.
  if (llvm::is_contained(tensorType.getShape(), 0)) {
    return op->emitError() << "tensor dimension size of zero is not allowed "
                              "with sub-channel quantization";
  }

  auto scaleShape =
      uniformQuantizedSubChannelType.getScales().getType().getShape();
  if (scaleShape.size() != shape.size()) {
    return op->emitError() << "Rank of scales " << scaleShape.size()
                           << " must match "
                           << "the rank of the tensor " << shape.size();
  }

  for (auto [index, scaleDim] : llvm::enumerate(expectedScaleShape)) {
    if (expectedScaleShape[index] != ShapedType::kDynamic &&
        expectedScaleShape[index] != scaleShape[index])
      return op->emitError() << "dimension size " << scaleDim
                             << " of scales tensor at axis " << index
                             << " should match (tensor dimension at axis / "
                                "block sizes at axis) = "
                             << expectedScaleShape[index];
  }

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

  // Verify integrity of per-axis quantization information, if present.
  if (auto quantizedPerAxisType =
          dyn_cast<UniformQuantizedPerAxisType>(quantizedType)) {
    return verifyPerAxisQuantization(op, quantizedPerAxisType, containerType);
  }

  if (auto quantizedSubChannelType =
          dyn_cast<UniformQuantizedSubChannelType>(quantizedType)) {
    return verifySubChannelQuantization(op, quantizedSubChannelType,
                                        containerType);
  }

  // At this point the type is UniformQuantizedType
  return success();
}

struct QuantInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  /// All quant dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void QuantDialect::initialize() {
  addTypes<AnyQuantizedType, CalibratedQuantizedType, UniformQuantizedType,
           UniformQuantizedPerAxisType, UniformQuantizedSubChannelType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quant/IR/QuantOps.cpp.inc"
      >();
  detail::addBytecodeInterface(this);
  addInterfaces<QuantInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// DequantizeCastOp
//===----------------------------------------------------------------------===//

LogicalResult DequantizeCastOp::verify() {
  return verifyQuantizationOp(*this, getQuantizedType(), getFloatType(),
                              getInput().getType());
}

OpFoldResult DequantizeCastOp::fold(FoldAdaptor adaptor) {
  // Matches x -> quant.qcast -> quant.dcast -> y, replacing the quant.dcast op
  // with the value of x. Values x and y are guaranteed to be of the same type
  // in this pattern.
  auto srcQcastOp = getInput().getDefiningOp<QuantizeCastOp>();
  if (!srcQcastOp)
    return {};
  assert(srcQcastOp.getInput().getType() == getType());
  return srcQcastOp.getInput();
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

OpFoldResult QuantizeCastOp::fold(FoldAdaptor adaptor) {
  // Matches x -> quant.dcast -> quant.qcast -> y, replacing the quant.qcast op
  // with the value of x if the casts invert each other. Contrary to the folding
  // pattern in quant.dcast (i.e., x -> quant.qcast -> quant.dcast -> y), values
  // x and y are not guaranteed to be of the same type here, as they may use
  // different quantization parameters.
  auto srcDcastOp = getInput().getDefiningOp<DequantizeCastOp>();
  if (!srcDcastOp || srcDcastOp.getInput().getType() != getType())
    return {};
  return srcDcastOp.getInput();
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

LogicalResult StorageCastOp::verify() {
  auto quantizedType = getQuantizedType();
  auto integerType = getIntegerType();
  if (quantizedType.getStorageType() != integerType)
    return emitError(
        "storage type in quantized type expected to match integer type");

  // Verify integrity of per-axis quantization information, if available. While
  // the quantization type may appear in the input or the result, their tensor
  // shapes are guaranteed to be identical at this point.
  if (auto quantizedPerAxisType =
          dyn_cast<UniformQuantizedPerAxisType>(quantizedType)) {
    return verifyPerAxisQuantization(*this, quantizedPerAxisType,
                                     getInput().getType());
  }

  if (auto quantizedSunChannelType =
          dyn_cast<UniformQuantizedSubChannelType>(quantizedType)) {
    return verifySubChannelQuantization(*this, quantizedSunChannelType,
                                        getInput().getType());
  }

  // At this point the type is UniformQuantizedType
  return success();
}

OpFoldResult StorageCastOp::fold(FoldAdaptor adaptor) {
  // Matches x -> quant.scast -> quant.scast -> y, replacing the second
  // quant.scast with the value of x if the casts invert each other.
  auto srcScastOp = getInput().getDefiningOp<StorageCastOp>();
  if (!srcScastOp || srcScastOp.getInput().getType() != getType())
    return {};
  return srcScastOp.getInput();
}

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

} // namespace quant
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Quant/IR/QuantOps.cpp.inc"
