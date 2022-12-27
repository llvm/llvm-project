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

#ifndef MLIR_DIALECT_SCF_UTILS_AFFINECANONICALIZATIONUTILS_H_
#define MLIR_DIALECT_SCF_UTILS_AFFINECANONICALIZATIONUTILS_H_

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class AffineApplyOp;
class AffineMap;
class FlatAffineValueConstraints;
struct LogicalResult;
class Operation;
class OpFoldResult;
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
using LoopMatcherFn = function_ref<LogicalResult(
    Value, OpFoldResult &, OpFoldResult &, OpFoldResult &)>;

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

/// Attempt to canonicalize min/max operations by proving that their value is
/// bounded by the same lower and upper bound. In such cases, the operation can
/// be folded away.
///
/// Bounds are computed by FlatAffineValueConstraints. Invariants required for
/// finding/proving bounds should be supplied via `constraints`.
///
/// 1. Add dimensions for `op` and `opBound` (lower or upper bound of `op`).
/// 2. Compute an upper bound of `op` (in case of `isMin`) or a lower bound (in
///    case of `!isMin`) and bind it to `opBound`. SSA values that are used in
///    `op` but are not part of `constraints`, are added as extra symbols.
/// 3. For each result of `op`: Add result as a dimension `r_i`. Prove that:
///    * If `isMin`: r_i >= opBound
///    * If `isMax`: r_i <= opBound
///    If this is the case, ub(op) == lb(op).
/// 4. Replace `op` with `opBound`.
///
/// In summary, the following constraints are added throughout this function.
/// Note: `invar` are dimensions added by the caller to express the invariants.
/// (Showing only the case where `isMin`.)
///
///  invar |    op | opBound | r_i | extra syms... | const |           eq/ineq
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (various eq./ineq. constraining `invar`, added by the caller)
///    ... |     0 |       0 |   0 |             0 |   ... |               ...
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (various ineq. constraining `op` in terms of `op` operands (`invar` and
///    extra `op` operands "extra syms" that are not in `invar`)).
///    ... |    -1 |       0 |   0 |           ... |   ... |              >= 0
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (set `opBound` to `op` upper bound in terms of `invar` and "extra syms")
///    ... |     0 |      -1 |   0 |           ... |   ... |               = 0
///  ------+-------+---------+-----+---------------+-------+-------------------
///   (for each `op` map result r_i: set r_i to corresponding map result,
///    prove that r_i >= minOpUb via contradiction)
///    ... |     0 |       0 |  -1 |           ... |   ... |               = 0
///      0 |     0 |       1 |  -1 |             0 |    -1 |              >= 0
///
FailureOr<AffineApplyOp>
canonicalizeMinMaxOp(RewriterBase &rewriter, Operation *op, AffineMap map,
                     ValueRange operands, bool isMin,
                     FlatAffineValueConstraints constraints);

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

#endif // MLIR_DIALECT_SCF_UTILS_AFFINECANONICALIZATIONUTILS_H_
