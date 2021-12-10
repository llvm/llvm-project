//===- Utils.h - SCF dialect utilities --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various SCF utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_UTILS_H_
#define MLIR_DIALECT_SCF_UTILS_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
class FuncOp;
class Operation;
class OpBuilder;
class ValueRange;
class Value;
class AffineExpr;
class Operation;

namespace scf {
class IfOp;
class ForOp;
class ParallelOp;
} // namespace scf

/// Create a clone of `loop` with `newIterOperands` added as new initialization
/// values and `newYieldedValues` added as new yielded values. The returned
/// ForOp has `newYieldedValues.size()` new result values.  The `loop` induction
/// variable and `newIterOperands` are remapped to the new induction variable
/// and the new entry block arguments respectively.
///
/// Additionally, if `replaceLoopResults` is true, all uses of
/// `loop.getResults()` are replaced with the first `loop.getNumResults()`
/// return values respectively. This additional replacement is provided as a
/// convenience to update the consumers of `loop`, in the case e.g. when `loop`
/// is soon to be deleted.
///
/// Return the cloned loop.
///
/// This convenience function is useful to factorize common mechanisms related
/// to hoisting roundtrips to memory into yields. It does not perform any
/// legality checks.
///
/// Prerequisite: `newYieldedValues.size() == newYieldedValues.size()`.
scf::ForOp cloneWithNewYields(OpBuilder &b, scf::ForOp loop,
                              ValueRange newIterOperands,
                              ValueRange newYieldedValues,
                              bool replaceLoopResults = true);

/// Outline the then and/or else regions of `ifOp` as follows:
///  - if `thenFn` is not null, `thenFnName` must be specified and the `then`
///    region is inlined into a new FuncOp that is captured by the pointer.
///  - if `elseFn` is not null, `elseFnName` must be specified and the `else`
///    region is inlined into a new FuncOp that is captured by the pointer.
void outlineIfOp(OpBuilder &b, scf::IfOp ifOp, FuncOp *thenFn,
                 StringRef thenFnName, FuncOp *elseFn, StringRef elseFnName);

/// Get a list of innermost parallel loops contained in `rootOp`. Innermost parallel
/// loops are those that do not contain further parallel loops themselves.
bool getInnermostParallelLoops(Operation *rootOp,
                               SmallVectorImpl<scf::ParallelOp> &result);

/// Return the min/max expressions for `value` if it is an induction variable
/// from scf.for or scf.parallel loop.
/// if `loopFilter` is passed, the filter determines which loop to consider.
/// Other induction variables are ignored.
Optional<std::pair<AffineExpr, AffineExpr>>
getSCFMinMaxExpr(Value value, SmallVectorImpl<Value> &dims,
                 SmallVectorImpl<Value> &symbols,
                 llvm::function_ref<bool(Operation *)> loopFilter = nullptr);

} // namespace mlir
#endif // MLIR_DIALECT_SCF_UTILS_H_
