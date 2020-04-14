//===- LinalgTransforms.h - Linalg transformations as patterns --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LINALG_TRANSFORMS_LINALGTRANSFORMS_H_
#define DIALECT_LINALG_TRANSFORMS_LINALGTRANSFORMS_H_

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace linalg {

// Marker used as attribute name in generated Linalg rewriting transformations.
struct LinalgTransforms {
  static const StringLiteral kLinalgTransformMarker;
};

namespace detail {
// Implementation detail of isProducedByOpOfType avoids the need for explicit
// template instantiations.
bool isProducedByOpOfTypeImpl(Operation *consumerOp, Value consumedView,
                              function_ref<bool(Operation *)> isaOpType);
} // namespace detail

// Returns true if the `consumedView` value use in `consumerOp` is produced by
// an op of type `OpTy`. This is used to implement use-def type information on
// buffers.
template <typename OpTy>
bool isProducedByOpOfType(Operation *consumerOp, Value consumedView) {
  return detail::isProducedByOpOfTypeImpl(
      consumerOp, consumedView, [](Operation *op) { return isa<OpTy>(op); });
}

////////////////////////////////////////////////////////////////////////////////
// The following Declarative Rewrite Rule (DRR) helpers are used in rewrite
// patterns. As such, they must not call into `rewriter.erase/replace` APIs and
// it is the responsibility of the enclosing PatternRewriter to erase on
// success.
////////////////////////////////////////////////////////////////////////////////

/// Tiles `op` by `sizes` permuting the loops according to `permutation` and
/// sets the attribute `kLinalgTransformMarker` to `linalgMarker`.  The
/// permutation is expressed as a list of integers that specify the new ordering
/// of the loop nest (using loop.for operations). The length of `permutation`
/// must be equal to the length of `tileSizes`.
/// E.g. the permutation `(i,j,k) -> (j,k,i)` will be expressed with
/// `permutation = [1,2,0]`. All values in `permutation` must be
/// integers, in the range 0..`tileSizes.size()` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation). An empty list
/// states for the identity permutation.
LogicalResult tileLinalgOpAndSetMarker(PatternRewriter &rewriter, Operation *op,
                                       ArrayRef<int64_t> sizes,
                                       StringRef linalgMarker,
                                       ArrayRef<unsigned> permutation);

/// Tiles ops similar to `tileLinalgOpAndSetMarker` but generates loop.parallel
/// operations instead.
LogicalResult tileLinalgOpToParallelLoopsAndSetMarker(
    PatternRewriter &rewriter, Operation *op, ArrayRef<int64_t> sizes,
    StringRef linalgMarker, ArrayRef<unsigned> permutation);

/// Tiles `op` by `sizes`, fuses the producers of `operandIndicesToFuse` and
/// sets the attribute `kLinalgTransformMarker` to `linalgMarker`.
LogicalResult tileAndFuseLinalgOpAndSetMarker(
    PatternRewriter &rewriter, Operation *op, ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> operandIndicesToFuse, StringRef linalgMarker);

/// Tiles ops similar to `tileAndFuseLinalgOpAndSetMarker` but generates
/// loop.parallel operations instead.
LogicalResult tileAndFuseLinalgOpToParallelLoopsAndSetMarker(
    PatternRewriter &rewriter, Operation *op, ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> operandIndicesToFuse, StringRef linalgMarker);

using LinalgLoops = SmallVector<Operation *, 4>;

/// Emits a loop nest of with the proper body for `op`.
template <typename LoopTy, typename ConcreteOp>
Optional<LinalgLoops> linalgLowerOpToLoops(PatternRewriter &rewriter,
                                           Operation *op);

/// Emits a loop nest of `loop.for` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult linalgOpToLoops(PatternRewriter &rewriter, Operation *op);

/// Emits a loop nest of `loop.parallel` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult linalgOpToParallelLoops(PatternRewriter &rewriter, Operation *op);

/// Emits a loop nest of `affine.for` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult linalgOpToAffineLoops(PatternRewriter &rewriter, Operation *op);

/// Rewrite a linalg.generic into a suitable vector.contraction op.
LogicalResult vectorizeLinalgOpPrecondition(Operation *op);
SmallVector<Value, 0> vectorizeLinalgOp(PatternRewriter &rewriter,
                                        Operation *op);

/// Emits a `generic` or `indexed_generic` operation with the `indexing_maps`
/// and `iterator_types` permutated according to `permutation`.
LogicalResult
permuteGenericLinalgOpPrecondition(Operation *op,
                                   ArrayRef<unsigned> permutation);
SmallVector<Value, 0> permuteGenericLinalgOp(PatternRewriter &rewriter,
                                             Operation *op,
                                             ArrayRef<unsigned> permutation,
                                             StringRef linalgMarker);

/// Promote std.subviews feeding linalg operations.
LogicalResult promoteSubviewsLinalgOpPrecondition(Operation *op);
SmallVector<Value, 0> promoteSubviewsLinalgOp(PatternRewriter &rewriter,
                                              Operation *op);

} // namespace linalg
} // namespace mlir

#endif // DIALECT_LINALG_TRANSFORMS_LINALGTRANSFORMS_H_
