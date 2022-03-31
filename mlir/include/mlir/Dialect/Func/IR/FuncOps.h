//===- FuncOps.h - Func Dialect Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_IR_OPS_H
#define MLIR_DIALECT_FUNC_IR_OPS_H

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class PatternRewriter;
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Func/IR/FuncOps.h.inc"

#include "mlir/Dialect/Func/IR/FuncOpsDialect.h.inc"

namespace mlir {
/// FIXME: This is a temporary using directive to ease the transition of FuncOp
/// to the Func dialect. This will be removed after all uses are updated.
using FuncOp = func::FuncOp;
} // namespace mlir

namespace llvm {

/// Allow stealing the low bits of FuncOp.
template <>
struct PointerLikeTypeTraits<mlir::func::FuncOp> {
  static inline void *getAsVoidPointer(mlir::func::FuncOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline mlir::func::FuncOp getFromVoidPointer(void *p) {
    return mlir::func::FuncOp::getFromOpaquePointer(p);
  }
  static constexpr int numLowBitsAvailable = 3;
};
} // namespace llvm

#endif // MLIR_DIALECT_FUNC_IR_OPS_H
