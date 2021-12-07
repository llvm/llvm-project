//===- AffineCanonicalizationUtils.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions to canonicalize affine ops
// within SCF op regions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_AFFINECANONICALIZATIONUTILS_H_
#define MLIR_DIALECT_SCF_AFFINECANONICALIZATIONUTILS_H_

#include "mlir/Support/LLVM.h"

namespace mlir {
class AffineMap;
struct LogicalResult;
class Operation;
class RewriterBase;
class Value;
class ValueRange;

namespace scf {
class IfOp;

/// Match "for loop"-like operations: If the first parameter is an iteration
/// variable, return lower/upper bounds via the second/third parameter and the
/// step size via the last parameter. The function should return `success` in
/// that case. If the first parameter is not an iteration variable, return
/// `failure`.
using LoopMatcherFn =
    function_ref<LogicalResult(Value, Value &, Value &, Value &)>;

/// Try to canonicalize an min/max operations in the context of for `loops` with
/// a known range.
///
/// `map` is the body of the min/max operation and `operands` are the SSA values
/// that the dimensions and symbols are bound to; dimensions are listed first.
/// If `isMin`, the operation is a min operation; otherwise, a max operation.
/// `loopMatcher` is used to retrieve loop bounds and the step size for a given
/// iteration variable.
///
/// Note: `loopMatcher` allows this function to be used with any "for loop"-like
/// operation (scf.for, scf.parallel and even ops defined in other dialects).
LogicalResult canonicalizeMinMaxOpInLoop(RewriterBase &rewriter, Operation *op,
                                         AffineMap map, ValueRange operands,
                                         bool isMin, LoopMatcherFn loopMatcher);

/// Try to simplify a min/max operation `op` after loop peeling. This function
/// can simplify min/max operations such as (ub is the previous upper bound of
/// the unpeeled loop):
/// ```
/// #map = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
/// %r = affine.min #affine.min #map(%iv)[%step, %ub]
/// ```
/// and rewrites them into (in the case the peeled loop):
/// ```
/// %r = %step
/// ```
/// min/max operations inside the partial iteration are rewritten in a similar
/// way.
LogicalResult rewritePeeledMinMaxOp(RewriterBase &rewriter, Operation *op,
                                    AffineMap map, ValueRange operands,
                                    bool isMin, Value iv, Value ub, Value step,
                                    bool insideLoop);

} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_AFFINECANONICALIZATIONUTILS_H_
