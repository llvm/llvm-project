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
#include "mlir/Dialect/Traits.h"
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

#include "mlir/Dialect/Tosa/IR/TosaOpsDialect.h.inc"

namespace mlir {
class PatternRewriter;

namespace tosa {

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
    auto resElemType = getElementTypeOrSelf(op->getResult(0));

    // In cases of floating point type, op requires the same element
    // type for all operands and result.
    if (llvm::isa<FloatType>(resElemType))
      return impl::verifySameOperandsAndResultElementType(op);

    if (auto resIntType = resElemType.dyn_cast<IntegerType>()) {
      IntegerType lhsIntType =
          getElementTypeOrSelf(op->getOperand(0)).cast<IntegerType>();
      IntegerType rhsIntType =
          getElementTypeOrSelf(op->getOperand(1)).cast<IntegerType>();
      if (lhsIntType != rhsIntType)
        return op->emitOpError(
            "requires the same element type for all operands");

      // Though the spec requires the element type of result to be i32, a more
      // relaxed way is provided at dialect level for easier cooperating with
      // other dialects.
      if (lhsIntType.getWidth() > resIntType.getWidth())
        return op->emitOpError("invalid data type size for operands or result");

      return success();
    }

    return failure();
  }
};

} // namespace tosa
} // namespace OpTrait

} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.h.inc"

#endif // MLIR_DIALECT_TOSA_IR_TOSAOPS_H
