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

// Create a pad-const const tensor with value of `val` of required data-type
Value createPadConstTensor(OpBuilder &builder, Location loc, Value src,
                           int32_t val = 0);

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_IR_TOSAOPS_H
