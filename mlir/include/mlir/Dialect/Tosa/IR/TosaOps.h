//===-- TosaOps.h - TOSA dialect operation definitions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_IR_TOSAOPS_H
#define MLIR_DIALECT_TOSA_IR_TOSAOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// TOSA dialect and structs includes.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaEnums.h.inc"
#include "mlir/Dialect/Tosa/IR/TosaOpsDialect.h.inc"
#include "mlir/Transforms/DialectConversion.h"

//===----------------------------------------------------------------------===//
// TOSA operation validation includes.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaAvailability.h.inc"

namespace mlir {
class PatternRewriter;

namespace tosa {

ParseResult parseTypeOrAttr(OpAsmParser &parser, TypeAttr &typeAttr,
                            Attribute &attr);
void printTypeOrAttr(OpAsmPrinter &p, Operation *op, TypeAttr type,
                     Attribute attr);

#include "mlir/Dialect/Tosa/IR/TosaInterfaces.h.inc"

} // namespace tosa

namespace OpTrait {
namespace tosa {

// This trait verifies if the element type amoung operands and result
// of multiplication match tosa specification.
template <typename ConcreteType>
class MulOperandsAndResultElementType
    : public TraitBase<ConcreteType, MulOperandsAndResultElementType> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // Check we have a single result.
    if (failed(impl::verifyOneResult(op)))
      return failure();
    Type resElemType = getElementTypeOrSelf(op->getResult(0));

    // Check we have lhs and rhs.
    if (failed(impl::verifyAtLeastNOperands(op, 2)))
      return failure();

    Type lhsElemType = getElementTypeOrSelf(op->getOperand(0));
    Type rhsElemType = getElementTypeOrSelf(op->getOperand(1));

    // Check that for i32 a shift has been explicitly provided.
    if (lhsElemType.isInteger(32) && failed(impl::verifyNOperands(op, 3)))
      return failure();

    // Verify operands type match (ignoring the shift parameter which will
    // always be i8).
    if (lhsElemType != rhsElemType)
      return op->emitOpError("requires the same element type for all operands");

    // Though the spec requires the element type of result to be i32, a more
    // relaxed way is provided at dialect level for easier cooperating with
    // other dialects.
    if (auto resIntType = dyn_cast<IntegerType>(resElemType)) {
      auto lhsIntType = cast<IntegerType>(lhsElemType);
      if (lhsIntType.getWidth() > resIntType.getWidth())
        return op->emitOpError("invalid data type size for operands or result");
    } else {
      // In cases of floating point type or quant types, op requires the same
      // element type for all operands and result (excluding shift).
      if (resElemType != lhsElemType)
        return op->emitOpError(
            "requires the same element type for all operands and results");
    }

    return llvm::success();
  }
};

/// This class indicates that an op is tosa-elementwise (permits broadcasting,
/// unlike Elementwise trait).
template <typename ConcreteType>
class TosaElementwiseOperator
    : public TraitBase<ConcreteType, TosaElementwiseOperator> {};

LogicalResult verifyTosaResolvableShapeOperands(Operation *op);
/// This class verifies that tosa shape operands are compile time resolvable
template <typename ConcreteType>
class TosaResolvableShapeOperands
    : public TraitBase<ConcreteType, TosaResolvableShapeOperands> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyTosaResolvableShapeOperands(op);
  }
};

LogicalResult verifyTosaShapeOperator(Operation *op);
/// This class indicates that op operates on tosa shape types
template <typename ConcreteType>
class TosaShapeOperator : public TraitBase<ConcreteType, TosaShapeOperator> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyTosaShapeOperator(op);
  }
};

LogicalResult verifyTosaShapeOperatorWithSameRanks(Operation *op);
/// This class indicates that op operates on tosa shape types
template <typename ConcreteType>
class TosaShapeOperatorWithSameRanks
    : public TraitBase<ConcreteType, TosaShapeOperatorWithSameRanks> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyTosaShapeOperatorWithSameRanks(op);
  }
};

} // namespace tosa
} // namespace OpTrait

namespace tosa {

bool isa_tosa_shape_type(mlir::Type t);

} // namespace tosa

} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOpsTypesBase.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.h.inc"

namespace mlir {
namespace tosa {

// Create a rank-1 const tensor for zero point of the source tensor.
std::optional<Value> createZeroPointTensor(OpBuilder &builder, Location loc,
                                           Type srcElemType, int64_t zp = 0);

// Get zero point value from the attribute argument.
LogicalResult getZeroPoint(ElementsAttr zpAttr, int64_t &zp);

// Verify if zero point falls into valid range.
template <typename T>
LogicalResult verifyZeroPoint(Type zpElemType, int64_t zp) {
  if constexpr (!std::is_same_v<T, Conv2DOp> && !std::is_same_v<T, Conv3DOp> &&
                !std::is_same_v<T, DepthwiseConv2DOp> &&
                !std::is_same_v<T, TransposeConv2DOp>) {
    return failure();
  }

  if (!zpElemType.isIntOrFloat())
    return failure();

  if (!zpElemType.isInteger(8) && zp != 0)
    return failure();

  if (zpElemType.isSignedInteger(8) && (zp < -128 || zp > 127))
    return failure();

  if (zpElemType.isUnsignedInteger(8) && (zp < 0 || zp > 255))
    return failure();

  return success();
}

// Helper type trait to determine if an operation is a tosa convolution.
template <typename Op>
struct IsTosaConv : std::false_type {};

template <>
struct IsTosaConv<tosa::Conv2DOp> : std::true_type {};
template <>
struct IsTosaConv<tosa::DepthwiseConv2DOp> : std::true_type {};
template <>
struct IsTosaConv<tosa::TransposeConv2DOp> : std::true_type {};
template <>
struct IsTosaConv<tosa::Conv3DOp> : std::true_type {};

template <typename Op>
constexpr bool is_tosa_conv_v = IsTosaConv<Op>::value;

// Helper struct to hold the zero points of a TOSA convolution operation as
// named 64-bit integer fields.
struct ConvZpPair {
  ConvZpPair(std::int64_t inputZp, std::int64_t weightZp)
      : inputZp(inputZp), weightZp(weightZp) {}
  std::int64_t inputZp;
  std::int64_t weightZp;
};

// Helper function which attempts to extract the zero points from a TOSA
// convolution by matching them against defining ops which should be tosa.const
// operations.
//
// There are three possible results:
// 1. Failed to extract the zero-points i.e. they should exist and don't or they
// do exist but are invalid.
// 2. Succeeded in extracting zero-points.
// 3. Zero points are "empty" and meaningless for this op i.e. non-quantized
// convolution.
using FailOrMaybeZP = llvm::FailureOr<std::optional<ConvZpPair>>;
template <typename TosaConvOp>
std::enable_if_t<is_tosa_conv_v<TosaConvOp>, FailOrMaybeZP>
extractConvZpPair(TosaConvOp op, PatternRewriter &rewriter) {
  // Strictly speaking the base TOSA spec requires that for non int8 types
  // zero points must be zero. However, in the dialect these operands are
  // optional and only required for int8. They have no semantic meaning for
  // non-quantized types and can therefore be safely ignored. This is case 3.
  if (auto opElementTY =
          cast<ShapedType>(op->getOperand(0).getType()).getElementType();
      !opElementTY.isInteger(8))
    return FailOrMaybeZP(std::nullopt);

  // Now we know we should have a zero point check it is valid.
  if (!op.getInputZp())
    return rewriter.notifyMatchFailure(op, "missing input zero point");

  // Helper to extract the zero point by matching its definition against a
  // constant.
  auto extractZeroPoint = [](Value zpValue) -> std::optional<int64_t> {
    ElementsAttr zpAttr;
    if (!matchPattern(zpValue, m_Constant(&zpAttr)))
      return std::nullopt;

    int64_t zp;
    if (tosa::getZeroPoint(zpAttr, zp).failed())
      return std::nullopt;

    return std::make_optional(zp);
  };

  auto maybeInputZp = extractZeroPoint(op.getInputZp());
  if (!maybeInputZp)
    return rewriter.notifyMatchFailure(op, "unable to extract input zp");

  if (!op.getWeightZp())
    return rewriter.notifyMatchFailure(op, "missing weight zero point");

  auto maybeWeightZp = extractZeroPoint(op.getWeightZp());
  if (!maybeWeightZp)
    return rewriter.notifyMatchFailure(op, "unable to extract weight zp");

  return std::make_optional<ConvZpPair>(*maybeInputZp, *maybeWeightZp);
}
} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_IR_TOSAOPS_H
