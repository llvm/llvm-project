//===- LoopAnalysis.h - loop analysis methods -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods to analyze loops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_LOOP_ANALYSIS_H
#define MLIR_ANALYSIS_LOOP_ANALYSIS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

namespace mlir {

class AffineExpr;
class AffineForOp;
class AffineMap;
class BlockArgument;
class MemRefType;
class NestedPattern;
class Operation;
class Value;

/// Returns the trip count of the loop as an affine map with its corresponding
/// operands if the latter is expressible as an affine expression, and nullptr
/// otherwise. This method always succeeds as long as the lower bound is not a
/// multi-result map. The trip count expression is simplified before returning.
/// This method only utilizes map composition to construct lower and upper
/// bounds before computing the trip count expressions
void getTripCountMapAndOperands(AffineForOp forOp, AffineMap *map,
                                SmallVectorImpl<Value> *operands);

/// Returns the trip count of the loop if it's a constant, None otherwise. This
/// uses affine expression analysis and is able to determine constant trip count
/// in non-trivial cases.
Optional<uint64_t> getConstantTripCount(AffineForOp forOp);

/// Returns the greatest known integral divisor of the trip count. Affine
/// expression analysis is used (indirectly through getTripCount), and
/// this method is thus able to determine non-trivial divisors.
uint64_t getLargestDivisorOfTripCount(AffineForOp forOp);

/// Given an induction variable `iv` of type AffineForOp and `indices` of type
/// IndexType, returns the set of `indices` that are independent of `iv`.
///
/// Prerequisites (inherited from `isAccessInvariant` above):
///   1. `iv` and `indices` of the proper type;
///   2. at most one affine.apply is reachable from each index in `indices`;
///
/// Emits a note if it encounters a chain of affine.apply and conservatively
///  those cases.
DenseSet<Value, DenseMapInfo<Value>>
getInvariantAccesses(Value iv, ArrayRef<Value> indices);

using VectorizableLoopFun = std::function<bool(AffineForOp)>;

/// Checks whether the loop is structurally vectorizable; i.e.:
///   1. no conditionals are nested under the loop;
///   2. all nested load/stores are to scalar MemRefs.
/// TODO: relax the no-conditionals restriction
bool isVectorizableLoopBody(AffineForOp loop,
                            NestedPattern &vectorTransferMatcher);

/// Checks whether the loop is structurally vectorizable and that all the LoadOp
/// and StoreOp matched have access indexing functions that are are either:
///   1. invariant along the loop induction variable created by 'loop';
///   2. varying along at most one memory dimension. If such a unique dimension
///      is found, it is written into `memRefDim`.
bool isVectorizableLoopBody(AffineForOp loop, int *memRefDim,
                            NestedPattern &vectorTransferMatcher);

/// Checks where SSA dominance would be violated if a for op's body
/// operations are shifted by the specified shifts. This method checks if a
/// 'def' and all its uses have the same shift factor.
// TODO: extend this to check for memory-based dependence violation when we have
// the support.
bool isOpwiseShiftValid(AffineForOp forOp, ArrayRef<uint64_t> shifts);

/// Utility to match a generic reduction given a list of iteration-carried
/// arguments, `iterCarriedArgs` and the position of the potential reduction
/// argument within the list, `redPos`. If a reduction is matched, returns the
/// reduced value and the topologically-sorted list of combiner operations
/// involved in the reduction. Otherwise, returns a null value.
///
/// The matching algorithm relies on the following invariants, which are subject
/// to change:
///  1. The first combiner operation must be a binary operation with the
///     iteration-carried value and the reduced value as operands.
///  2. The iteration-carried value and combiner operations must be side
///     effect-free, have single result and a single use.
///  3. Combiner operations must be immediately nested in the region op
///     performing the reduction.
///  4. Reduction def-use chain must end in a terminator op that yields the
///     next iteration/output values in the same order as the iteration-carried
///     values in `iterCarriedArgs`.
///  5. `iterCarriedArgs` must contain all the iteration-carried/output values
///     of the region op performing the reduction.
///
/// This utility is generic enough to detect reductions involving multiple
/// combiner operations (disabled for now) across multiple dialects, including
/// Linalg, Affine and SCF. For the sake of genericity, it does not return
/// specific enum values for the combiner operations since its goal is also
/// matching reductions without pre-defined semantics in core MLIR. It's up to
/// each client to make sense out of the list of combiner operations. It's also
/// up to each client to check for additional invariants on the expected
/// reductions not covered by this generic matching.
Value matchReduction(ArrayRef<BlockArgument> iterCarriedArgs, unsigned redPos,
                     SmallVectorImpl<Operation *> &combinerOps);
} // end namespace mlir

#endif // MLIR_ANALYSIS_LOOP_ANALYSIS_H
