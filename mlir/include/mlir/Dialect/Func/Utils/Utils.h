//===- Utils.h - General Func transformation utilities ----*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the Func dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_UTILS_H
#define MLIR_DIALECT_FUNC_UTILS_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

namespace func {

class FuncOp;
class CallOp;

/// Creates a new function operation with the same name as the original
/// function operation, but with the arguments reordered according to
/// the `newArgsOrder` and `newResultsOrder`.
/// The `funcOp` operation must have exactly one block.
/// Returns the new function operation or failure if `funcOp` doesn't
/// have exactly one block.
FailureOr<FuncOp>
replaceFuncWithNewOrder(RewriterBase &rewriter, FuncOp funcOp,
                        llvm::ArrayRef<unsigned> newArgsOrder,
                        llvm::ArrayRef<unsigned> newResultsOrder);
/// Creates a new call operation with the values as the original
/// call operation, but with the arguments reordered according to
/// the `newArgsOrder` and `newResultsOrder`.
CallOp replaceCallOpWithNewOrder(RewriterBase &rewriter, CallOp callOp,
                                 llvm::ArrayRef<unsigned> newArgsOrder,
                                 llvm::ArrayRef<unsigned> newResultsOrder);

} // namespace func
} // namespace mlir

#endif // MLIR_DIALECT_FUNC_UTILS_H
