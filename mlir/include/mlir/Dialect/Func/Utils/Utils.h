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
/// function operation, but with the arguments mapped according to
/// the `oldArgToNewArg` and `oldResToNewRes`.
/// The `funcOp` operation must have exactly one block.
/// Returns the new function operation or failure if `funcOp` doesn't
/// have exactly one block.
/// Note: the method asserts that the `oldArgToNewArg` and `oldResToNewRes`
/// maps the whole function arguments and results.
mlir::FailureOr<mlir::func::FuncOp> replaceFuncWithNewMapping(
    mlir::RewriterBase &rewriter, mlir::func::FuncOp funcOp,
    ArrayRef<int> oldArgIdxToNewArgIdx, ArrayRef<int> oldResIdxToNewResIdx);
/// Creates a new call operation with the values as the original
/// call operation, but with the arguments mapped according to
/// the `oldArgToNewArg` and `oldResToNewRes`.
/// Note: the method asserts that the `oldArgToNewArg` and `oldResToNewRes`
/// maps the whole call operation arguments and results.
mlir::func::CallOp replaceCallOpWithNewMapping(
    mlir::RewriterBase &rewriter, mlir::func::CallOp callOp,
    ArrayRef<int> oldArgIdxToNewArgIdx, ArrayRef<int> oldResIdxToNewResIdx);

} // namespace func
} // namespace mlir

#endif // MLIR_DIALECT_FUNC_UTILS_H
