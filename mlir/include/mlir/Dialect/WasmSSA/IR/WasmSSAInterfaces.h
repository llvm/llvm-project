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
LogicalResult verifyConstantExpressionInterface(Operation *op);
LogicalResult verifyLabelBranchingOpInterface(Operation *op);
template <typename OpType>
LogicalResult verifyLabelLevelInterfaceIsTerminator() {
  static_assert(OpType::template hasTrait<::mlir::OpTrait::IsTerminator>(),
                "LabelLevelOp should be terminator ops");
  return success();
}
LogicalResult verifyLabelLevelInterface(Operation *op);
} // namespace detail
template <class OperationType>
struct AlwaysValidConstantExprOpTrait
    : public OpTrait::TraitBase<OperationType, AlwaysValidConstantExprOpTrait> {};

template<typename OpType>
struct ConstantExpressionInitializerOpTrait : public OpTrait::TraitBase<OpType, ConstantExpressionInitializerOpTrait>{
    static LogicalResult verifyTrait(Operation* op) {
        return detail::verifyConstantExpressionInterface(op);
    }
};

} // namespace mlir::wasmssa
#include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.h.inc"

#endif // MLIR_DIALECT_WasmSSA_IR_WasmSSAINTERFACES_H_
