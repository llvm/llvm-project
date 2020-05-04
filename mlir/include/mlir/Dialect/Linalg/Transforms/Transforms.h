//===- Transforms.h - Linalg transformations as patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H_
#define DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// Transformations exposed as function calls.
//===----------------------------------------------------------------------===//
using LinalgLoops = SmallVector<Operation *, 4>;

struct TiledLinalgOp {
  LinalgOp op;
  SmallVector<Operation *, 8> loops;
};

/// Performs standalone tiling of a single LinalgOp by `tileSizes`.
/// and permute the loop nest according to `interchangeVector`
/// The permutation is expressed as a list of integers that specify
/// the new ordering of the loop nest. The length of `interchangeVector`
/// must be equal to the length of `tileSizes`.
/// An empty vector is interpreted as the identity permutation and the
/// transformation returns early.
///
/// When non-null, the optional pointer `folder` is used to call into the
/// `createAndFold` builder method. If `folder` is null, the regular `create`
/// method is called.
///
/// Returns a struct containing the tiled loops in the specified order
/// and the cloned op if successful, llvm::None otherwise.
///
/// E.g. the permutation `(i,j,k) -> (j,k,i)` is expressed by
/// `interchangeVector = [1,2,0]`. All values in `interchangeVector` must be
/// integers, in the range 0..`tileSizes.size()` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation).
Optional<TiledLinalgOp> tileLinalgOp(OpBuilder &b, LinalgOp op,
                                     ArrayRef<Value> tileSizes,
                                     ArrayRef<unsigned> interchangeVector = {},
                                     OperationFolder *folder = nullptr);
Optional<TiledLinalgOp>
tileLinalgOpToParallelLoops(OpBuilder &b, LinalgOp op,
                            ArrayRef<Value> tileSizes,
                            ArrayRef<unsigned> interchangeVector = {},
                            OperationFolder *folder = nullptr);

/// Performs standalone tiling of a single LinalgOp by constant `tileSizes`.
/// See `tileLinalgOp(... ArrayRef<Value> tileSizes,)` for more details
Optional<TiledLinalgOp> tileLinalgOp(OpBuilder &b, LinalgOp op,
                                     ArrayRef<int64_t> tileSizes,
                                     ArrayRef<unsigned> interchangeVector = {},
                                     OperationFolder *folder = nullptr);
Optional<TiledLinalgOp>
tileLinalgOpToParallelLoops(OpBuilder &b, LinalgOp op,
                            ArrayRef<int64_t> tileSizes,
                            ArrayRef<unsigned> interchangeVector = {},
                            OperationFolder *folder = nullptr);

/// Interchanges the `iterator_types` and `iterator_maps` dimensions of `op`.
/// This is an in-place transformation controlled by `interchangeVector`.
/// An empty vector is interpreted as the identity permutation and the
/// transformation returns early.
///
/// E.g. the permutation `(i,j,k) -> (j,k,i)` is expressed with
/// `interchangeVector = [1,2,0]`. All values in `interchangeVector` must be
/// integers, in the range 0..`op.rank` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation).
LinalgOp interchange(LinalgOp op, ArrayRef<unsigned> interchangeVector);

/// Promotes the `subViews` into a new buffer allocated at the insertion point
/// `b`. Promotion occurs in 3 steps:
///   1. Create a new buffer for a full tile (i.e. not clipped at the boundary).
///   2. Take a full view on the buffer and `linalg.fill` it with zeros (use
///      float zero for now).
///   3. Take a partial slice of the full view in step 2. and copy into it.
/// Infers statically sized buffers from subViews unless `dynamicBuffers` is
/// true.
///
/// Returns a list of PromotionInfo which hold the promoted buffer and the
/// full and partial views indexing into the buffer.
// TODO: revisit dynamicBuffers option.
LinalgOp promoteSubViewOperands(OpBuilder &b, LinalgOp op,
                                llvm::SetVector<Value> subViews,
                                bool dynamicBuffers = false,
                                int64_t alignment = 0,
                                OperationFolder *folder = nullptr);

/// Emit a suitable vector form for a Linalg op with fully static shape.
void vectorizeLinalgOp(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `LoopTy` with the proper body for `op`.
template <typename LoopTy, typename ConcreteOp>
Optional<LinalgLoops> linalgLowerOpToLoops(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `loop.for` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult linalgOpToLoops(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `loop.parallel` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult linalgOpToParallelLoops(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `affine.for` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult linalgOpToAffineLoops(OpBuilder &builder, Operation *op);

//===----------------------------------------------------------------------===//
// Preconditions that ensure the corresponding transformation suceeds and can be
// applied as a rewrite pattern.
//===----------------------------------------------------------------------===//
/// Emits a `generic` or `indexed_generic` operation with the `indexing_maps`
/// and `iterator_types` permutated according to `permutation`.
LogicalResult
interchangeGenericLinalgOpPrecondition(Operation *op,
                                       ArrayRef<unsigned> interchangeVector);

/// Promote std.subviews feeding linalg operations.
LogicalResult promoteSubviewsLinalgOpPrecondition(
    Operation *op, Optional<DenseSet<unsigned>> operandIndicesToPromote = None);

/// Rewrite a linalg.generic into a suitable vector.contraction op.
LogicalResult vectorizeLinalgOpPrecondition(Operation *op);

//===----------------------------------------------------------------------===//
// Transformations exposed as rewrite patterns.
//===----------------------------------------------------------------------===//
// Marker used as attribute name in generated Linalg rewriting transformations.
struct LinalgTransforms {
  static const StringLiteral kLinalgTransformMarker;
};

/// Helper class to control common attribute matching and setting behavior.
struct LinalgMarker {
  LinalgMarker(ArrayRef<StringRef> matchDisjunction = {},
               Optional<StringRef> replacement = None);
  LinalgMarker(ArrayRef<StringRef> matchDisjunction, StringRef replacement);
  LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
  void replaceLinalgMarker(PatternRewriter &rewriter, Operation *op) const;

private:
  SmallVector<StringRef, 4> matchDisjunction;
  Optional<StringRef> replacement;
};

///
/// Linalg tiling patterns.
///
/// Apply the `tileLinalgOp` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `tileLinalgOp` for more details.
enum class LinalgTilingLoopType {
  Loops = 0,
  AffineLoops = 1,
  ParallelLoops = 2
};
struct LinalgTilingOptions {
  /// The tile sizes by which to tile.
  SmallVector<int64_t, 4> tileSizes{};
  LinalgTilingOptions &setTileSizes(ArrayRef<int64_t> ts) {
    tileSizes.assign(ts.begin(), ts.end());
    return *this;
  }
  /// The interchange vector to reorder the tiled loops.
  SmallVector<unsigned, 4> interchangeVector{};
  LinalgTilingOptions &setInterchange(ArrayRef<unsigned> interchange) {
    interchangeVector.assign(interchange.begin(), interchange.end());
    return *this;
  }
  /// The type of tile loops to generate.
  LinalgTilingLoopType loopType{LinalgTilingLoopType::Loops};
  LinalgTilingOptions &setLoopType(LinalgTilingLoopType lt) {
    loopType = lt;
    return *this;
  }
};

struct LinalgBaseTilingPattern : public RewritePattern {
  LinalgBaseTilingPattern(StringRef opName, MLIRContext *context,
                          LinalgTilingOptions options,
                          LinalgMarker marker = LinalgMarker(),
                          PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
  /// Options to control tiling;
  LinalgTilingOptions options;
};

template <typename OpTy>
struct LinalgTilingPattern : public LinalgBaseTilingPattern {
  LinalgTilingPattern(MLIRContext *context, LinalgTilingOptions options,
                      LinalgMarker marker = LinalgMarker(),
                      PatternBenefit benefit = 1)
      : LinalgBaseTilingPattern(OpTy::getOperationName(), context, options,
                                marker, benefit) {}
};

///
/// Linalg interchange patterns.
///
/// Apply the `interchange` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `interchange` for more details.
struct LinalgBaseInterchangePattern : public RewritePattern {
  LinalgBaseInterchangePattern(StringRef opName, MLIRContext *context,
                               ArrayRef<unsigned> interchangeVector,
                               LinalgMarker marker = LinalgMarker(),
                               PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
  /// The interchange vector to reorder the iterators and indexing_maps dims.
  SmallVector<unsigned, 8> interchangeVector;
};

template <typename OpTy>
struct LinalgInterchangePattern : public LinalgBaseInterchangePattern {
  LinalgInterchangePattern(MLIRContext *context,
                           ArrayRef<unsigned> interchangeVector,
                           LinalgMarker marker = LinalgMarker(),
                           PatternBenefit benefit = 1)
      : LinalgBaseInterchangePattern(OpTy::getOperationName(), context,
                                     interchangeVector, marker, benefit) {}
};

///
/// Linalg promotion patterns.
///
/// Apply the `promoteSubViewOperands` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `promoteSubViewOperands` for more details.
struct LinalgBasePromotionPattern : public RewritePattern {
  LinalgBasePromotionPattern(StringRef opName, MLIRContext *context,
                             ArrayRef<unsigned> operandsToPromote = {},
                             unsigned alignment = 0,
                             LinalgMarker marker = LinalgMarker(),
                             PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
  /// Indices of subViews to promote.
  SmallVector<unsigned, 4> operandsToPromote;
  /// Alignment of promoted buffer.
  unsigned alignment;
};

template <typename OpTy>
struct LinalgPromotionPattern : public LinalgBasePromotionPattern {
  LinalgPromotionPattern(MLIRContext *context,
                         ArrayRef<unsigned> operandsToPromote = {},
                         unsigned alignment = 0,
                         LinalgMarker marker = LinalgMarker(),
                         PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(OpTy::getOperationName(), context,
                                   operandsToPromote, alignment, marker,
                                   benefit) {}
  LinalgPromotionPattern(MLIRContext *context,
                         ArrayRef<unsigned> operandsToPromote,
                         LinalgMarker marker = LinalgMarker(),
                         PatternBenefit benefit = 1)
      : LinalgPromotionPattern(context, operandsToPromote, 0, marker, benefit) {
  }
  LinalgPromotionPattern(MLIRContext *context, unsigned alignment,
                         LinalgMarker marker = LinalgMarker(),
                         PatternBenefit benefit = 1)
      : LinalgPromotionPattern(context, {}, alignment, marker, benefit) {}
  LinalgPromotionPattern(MLIRContext *context, LinalgMarker marker,
                         PatternBenefit benefit = 1)
      : LinalgPromotionPattern(context, {}, 0, marker, benefit) {}
};

///
/// Linalg vectorization patterns.
///
/// Apply the `vectorizeLinalgOp` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `vectorizeLinalgOp` for more details.
struct LinalgBaseVectorizationPattern : public RewritePattern {
  LinalgBaseVectorizationPattern(StringRef opName, MLIRContext *context,
                                 LinalgMarker marker = LinalgMarker(),
                                 PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
};

template <typename OpTy>
struct LinalgVectorizationPattern : public LinalgBaseVectorizationPattern {
  LinalgVectorizationPattern(MLIRContext *context,
                             LinalgMarker marker = LinalgMarker(),
                             PatternBenefit benefit = 1)
      : LinalgBaseVectorizationPattern(OpTy::getOperationName(), context,
                                       marker, benefit) {}
};

///
/// Linalg lowering patterns.
///
/// Apply the `linalgLowerOpToLoops` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `linalgLowerOpToLoops` for more details.
enum class LinalgLoweringType {
  LibraryCall = 0,
  Loops = 1,
  AffineLoops = 2,
  ParallelLoops = 3
};
template <typename OpTy>
struct LinalgLoweringPattern : public RewritePattern {
  LinalgLoweringPattern(MLIRContext *context, LinalgLoweringType loweringType,
                        LinalgMarker marker = LinalgMarker(),
                        PatternBenefit benefit = 1)
      : RewritePattern(OpTy::getOperationName(), {}, benefit, context),
        marker(marker), loweringType(loweringType) {}
  // TODO: Move implementation to .cpp once named ops are auto-generated.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
    if (!linalgOp)
      return failure();
    if (failed(marker.checkAndNotify(rewriter, linalgOp)))
      return failure();
    if (failed(promoteSubviewsLinalgOpPrecondition(op)))
      return failure();

    if (loweringType == LinalgLoweringType::LibraryCall) {
      // TODO: Move lowering to library calls here.
      return failure();
    } else if (loweringType == LinalgLoweringType::Loops) {
      if (failed(linalgOpToLoops<OpTy>(rewriter, op)))
        return failure();
    } else if (loweringType == LinalgLoweringType::AffineLoops) {
      if (failed(linalgOpToAffineLoops<OpTy>(rewriter, op)))
        return failure();
    } else if (failed(linalgOpToParallelLoops<OpTy>(rewriter, op))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
  /// Controls whether the pattern lowers to library calls, loop.for, affine.for
  /// or loop.parallel.
  LinalgLoweringType loweringType;
};

} // namespace linalg
} // namespace mlir

#endif // DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H_
