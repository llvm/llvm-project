//===- QuantOps.cpp - Quantization Type and Ops Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/QuantOps.h"
#include "TypeDetail.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

using namespace mlir;
using namespace mlir::quant;
using namespace mlir::quant::detail;

#include "mlir/Dialect/Quant/QuantOpsDialect.cpp.inc"

void QuantizationDialect::initialize() {
  addTypes<AnyQuantizedType, CalibratedQuantizedType, UniformQuantizedType,
           UniformQuantizedPerAxisType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quant/QuantOps.cpp.inc"
      >();
}

OpFoldResult StorageCastOp::fold(ArrayRef<Attribute> operands) {
  // Matches x -> [scast -> scast] -> y, replacing the second scast with the
  // value of x if the casts invert each other.
  auto srcScastOp = getArg().getDefiningOp<StorageCastOp>();
  if (!srcScastOp || srcScastOp.getArg().getType() != getType())
    return OpFoldResult();
  return srcScastOp.getArg();
}

/// The quantization specification should match the expressed type.
static bool isValidQuantizationSpec(Attribute quantSpec, Type expressed) {
  if (auto typeAttr = quantSpec.dyn_cast<TypeAttr>()) {
    Type spec = typeAttr.getValue();
    if (spec.isa<TensorType, VectorType>())
      return false;

    // The spec should be either a quantized type which is compatible to the
    // expressed type, or a primitive type which is as same as the
    // (element type of) the expressed type.
    if (auto quantizedType = spec.dyn_cast<QuantizedType>())
      return quantizedType.isCompatibleExpressedType(expressed);

    if (auto tensorType = expressed.dyn_cast<TensorType>())
      return spec == tensorType.getElementType();

    if (auto vectorType = expressed.dyn_cast<VectorType>())
      return spec == vectorType.getElementType();
  }
  return false;
}

LogicalResult QuantizeRegionOp::verify() {
  // There are specifications for both inputs and outputs.
  if (getNumOperands() != getInputSpecs().size() ||
      getNumResults() != getOutputSpecs().size())
    return emitOpError(
        "has unmatched operands/results number and spec attributes number");

  // Verify that quantization specifications are valid.
  for (auto input : llvm::zip(getOperandTypes(), getInputSpecs())) {
    Type inputType = std::get<0>(input);
    Attribute inputSpec = std::get<1>(input);
    if (!isValidQuantizationSpec(inputSpec, inputType)) {
      return emitOpError() << "has incompatible specification " << inputSpec
                           << " and input type " << inputType;
    }
  }

  for (auto result : llvm::zip(getResultTypes(), getOutputSpecs())) {
    Type outputType = std::get<0>(result);
    Attribute outputSpec = std::get<1>(result);
    if (!isValidQuantizationSpec(outputSpec, outputType)) {
      return emitOpError() << "has incompatible specification " << outputSpec
                           << " and output type " << outputType;
    }
  }
  return success();
}

LogicalResult StatisticsOp::verify() {
  auto tensorArg = getArg().getType().dyn_cast<TensorType>();
  if (!tensorArg)
    return emitOpError("arg needs to be tensor type.");

  // Verify layerStats attribute.
  {
    auto layerStatsType = getLayerStats().getType();
    if (!layerStatsType.getElementType().isa<FloatType>()) {
      return emitOpError("layerStats must have a floating point element type");
    }
    if (layerStatsType.getRank() != 1 || layerStatsType.getDimSize(0) != 2) {
      return emitOpError("layerStats must have shape [2]");
    }
  }
  // Verify axisStats (optional) attribute.
  if (getAxisStats()) {
    if (!getAxis())
      return emitOpError("axis must be specified for axisStats");

    auto shape = tensorArg.getShape();
    auto argSliceSize =
        std::accumulate(std::next(shape.begin(), *getAxis()), shape.end(), 1,
                        std::multiplies<int64_t>());

    auto axisStatsType = getAxisStats()->getType();
    if (!axisStatsType.getElementType().isa<FloatType>()) {
      return emitOpError("axisStats must have a floating point element type");
    }
    if (axisStatsType.getRank() != 2 || axisStatsType.getDimSize(1) != 2 ||
        axisStatsType.getDimSize(0) != argSliceSize) {
      return emitOpError("axisStats must have shape [N,2] "
                         "where N = the slice size defined by the axis dim");
    }
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Quant/QuantOps.cpp.inc"
