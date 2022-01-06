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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
class FuncOp;
class Location;
class Operation;
class OpBuilder;
class Region;
class RewriterBase;
class ValueRange;
class Value;

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

/// Outline a region with a single block into a new FuncOp.
/// Assumes the FuncOp result types is the type of the yielded operands of the
/// single block. This constraint makes it easy to determine the result.
/// This method also clones the `arith::ConstantIndexOp` at the start of
/// `outlinedFuncBody` to alloc simple canonicalizations.
/// Creates a new FuncOp and thus cannot be used in a FunctionPass.
/// The client is responsible for providing a unique `funcName` that will not
/// collide with another FuncOp name.
// TODO: support more than single-block regions.
// TODO: more flexible constant handling.
FailureOr<FuncOp> outlineSingleBlockRegion(RewriterBase &rewriter, Location loc,
                                           Region &region, StringRef funcName);

/// Outline the then and/or else regions of `ifOp` as follows:
///  - if `thenFn` is not null, `thenFnName` must be specified and the `then`
///    region is inlined into a new FuncOp that is captured by the pointer.
///  - if `elseFn` is not null, `elseFnName` must be specified and the `else`
///    region is inlined into a new FuncOp that is captured by the pointer.
/// Creates new FuncOps and thus cannot be used in a FunctionPass.
/// The client is responsible for providing a unique `thenFnName`/`elseFnName`
/// that will not collide with another FuncOp name.
LogicalResult outlineIfOp(RewriterBase &b, scf::IfOp ifOp, FuncOp *thenFn,
                          StringRef thenFnName, FuncOp *elseFn,
                          StringRef elseFnName);

/// Get a list of innermost parallel loops contained in `rootOp`. Innermost
/// parallel loops are those that do not contain further parallel loops
/// themselves.
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
