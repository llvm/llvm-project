//===- WasmSSAInterfaces.h - WasmSSA Interfaces ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op interfaces for the WasmSSA dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_WasmSSA_IR_WasmSSAINTERFACES_H_
#define MLIR_DIALECT_WasmSSA_IR_WasmSSAINTERFACES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::wasmssa {
namespace detail {
/// Verify that `op` conforms to the ConstantExpressionInterface.
/// `op` must be initialized with valid constant expressions.
LogicalResult verifyConstantExpressionInterface(Operation *op);

/// Verify that `op` conforms to the LabelBranchingOpInterface
/// Checks that the branching is targetting something within its scope.
LogicalResult verifyLabelBranchingOpInterface(Operation *op);

/// Verify that `op` conforms to LabelLevelInterfaceIsTerminator
template <typename OpType>
LogicalResult verifyLabelLevelInterfaceIsTerminator() {
  static_assert(OpType::template hasTrait<::mlir::OpTrait::IsTerminator>(),
                "LabelLevelOp should be terminator ops");
  return success();
}

/// Verify that `op` conforms to the LabelLevelInterface
/// `op`'s target should defined at the same scope level.
LogicalResult verifyLabelLevelInterface(Operation *op);
} // namespace detail

/// Operations implementing this trait are considered as valid
/// constant expressions in any context (In contrast of
/// ConstantExprCheckOpInterface which are sometimes considered valid constant
/// expressions.
template <class OperationType>
struct ConstantExprOpTrait
    : public OpTrait::TraitBase<OperationType, ConstantExprOpTrait> {};

/// Trait used to verify operations that need a constant expression initializer.
template <typename OpType>
struct ConstantExpressionInitializerOpTrait
    : public OpTrait::TraitBase<OpType, ConstantExpressionInitializerOpTrait> {
  static LogicalResult verifyTrait(Operation *op) {
    return detail::verifyConstantExpressionInterface(op);
  }
};

} // namespace mlir::wasmssa
#include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.h.inc"

#endif // MLIR_DIALECT_WasmSSA_IR_WasmSSAINTERFACES_H_
