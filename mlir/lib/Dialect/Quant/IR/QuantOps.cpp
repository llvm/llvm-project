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
  auto srcScastOp = arg().getDefiningOp<StorageCastOp>();
  if (!srcScastOp || srcScastOp.arg().getType() != getType())
    return OpFoldResult();
  return srcScastOp.arg();
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

static LogicalResult verifyRegionOp(QuantizeRegionOp op) {
  // There are specifications for both inputs and outputs.
  if (op.getNumOperands() != op.input_specs().size() ||
      op.getNumResults() != op.output_specs().size())
    return op.emitOpError(
        "has unmatched operands/results number and spec attributes number");

  // Verify that quantization specifications are valid.
  for (auto input : llvm::zip(op.getOperandTypes(), op.input_specs())) {
    Type inputType = std::get<0>(input);
    Attribute inputSpec = std::get<1>(input);
    if (!isValidQuantizationSpec(inputSpec, inputType)) {
      return op.emitOpError() << "has incompatible specification " << inputSpec
                              << " and input type " << inputType;
    }
  }

  for (auto result : llvm::zip(op.getResultTypes(), op.output_specs())) {
    Type outputType = std::get<0>(result);
    Attribute outputSpec = std::get<1>(result);
    if (!isValidQuantizationSpec(outputSpec, outputType)) {
      return op.emitOpError() << "has incompatible specification " << outputSpec
                              << " and output type " << outputType;
    }
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Quant/QuantOps.cpp.inc"
