//===- LinalgInterface.h - Linalg operations interfaces -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interfaces for Linalg operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_IR_LINALGINTERFACES_H_
#define MLIR_DIALECT_LINALG_IR_LINALGINTERFACES_H_

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace linalg {
class LinalgOp;

/// OpOperand vector that implicitly converts to a Value vector.
struct OpOperandVector : public SmallVector<OpOperand *> {
  operator SmallVector<Value>();
};

namespace detail {
/// Implementation of the method that that check if given operands
/// can be dropped, i.e. the remaining operands can compute the loop
/// bounds of the op.
bool canOpOperandsBeDroppedImpl(linalg::LinalgOp linalgOp,
                                ArrayRef<OpOperand *> droppedOperands);
} // namespace detail

/// Checks whether `linalgOp` conforms to ContractionOpInterface.
// TODO: embed within `isa<ContractionOpInterface>` if possible / natural.
bool isaContractionOpInterface(LinalgOp linalgOp);

namespace detail {

/// Verify that `op` conforms to ContractionOpInterface.
LogicalResult verifyContractionInterface(Operation *op);

/// Verify that `op` conforms to the ConvolutionOpInterface.
LogicalResult verifyConvolutionInterface(Operation *op);

/// Verify that `op` conforms to the FillOpInterface.
LogicalResult verifyFillInterface(Operation *op);

/// Verify that `op` conforms to the invariants of StructuredOpInterface
LogicalResult verifyStructuredOpInterface(Operation *op);

/// Verify that `op` conforms to the invariants of DestinationStyleOpInterface
LogicalResult verifyDestinationStyleOpInterface(Operation *op);

} // namespace detail
} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.h.inc"

/// Include the generated interface declarations.
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h.inc"

#endif // MLIR_DIALECT_LINALG_IR_LINALGINTERFACES_H_
