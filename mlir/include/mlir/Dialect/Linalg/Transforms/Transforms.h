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
#include "llvm/ADT/SmallBitVector.h"

namespace mlir {
namespace linalg {

struct LinalgTilingOptions;

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
/// Returns a struct containing the tiled loops in the specified order
/// and the cloned op if successful, llvm::None otherwise.
///
/// E.g. the permutation `(i,j,k) -> (j,k,i)` is expressed by
/// `interchangeVector = [1,2,0]`. All values in `interchangeVector` must be
/// integers, in the range 0..`tileSizes.size()` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation).
Optional<TiledLinalgOp> tileLinalgOp(OpBuilder &b, LinalgOp op,
                                     const LinalgTilingOptions &options);

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

/// Callback function type used to perform the allocation for the promoted
/// `subView`. In `boundingSubViewsize` a best attempt is made to find the
/// smallest constant value for the size of the buffer needed for each
/// dimension. If that is not possible, contains the dynamic size of the
/// subview. The call back should return the buffer to use.
using AllocBufferCallbackFn = std::function<Optional<Value>(
    OpBuilder &b, SubViewOp subView, ArrayRef<Value> boundingSubViewSize,
    OperationFolder *folder)>;

/// Callback function type used to deallocate the buffers used to hold the
/// promoted subview.
using DeallocBufferCallbackFn =
    std::function<LogicalResult(OpBuilder &b, Value buffer)>;

/// Callback function type used to insert copy from original subview to subview
/// of the promoted region for the read operands/subview of promoted region to
/// original subview for the results. The copy has to happen from `src` to
/// `dst`.
using CopyCallbackFn =
    std::function<LogicalResult(OpBuilder &b, Value src, Value dst)>;

struct LinalgPromotionOptions {
  /// Indices of subViews to promote. If `None`, try to promote all operands.
  Optional<DenseSet<unsigned>> operandsToPromote = None;
  LinalgPromotionOptions &setOperandsToPromote(ArrayRef<int64_t> operands) {
    operandsToPromote = DenseSet<unsigned>();
    operandsToPromote->insert(operands.begin(), operands.end());
    return *this;
  }
  /// If ith element of `useFullTiles` is true the full view should be used for
  /// the promoted buffer of the ith operand in `operandsToPromote`. Otherwise
  /// the partial view will be used.
  /// The decision is defaulted to `useFullTileBuffersDefault` when
  /// `useFullTileBuffers` is None and for operands missing from
  /// `useFullTileBuffers`.
  Optional<llvm::SmallBitVector> useFullTileBuffers = None;
  LinalgPromotionOptions &setUseFullTileBuffers(ArrayRef<bool> useFullTiles) {
    unsigned size = useFullTiles.size();
    llvm::SmallBitVector tmp(size, false);
    for (unsigned i = 0; i < size; ++i)
      tmp[i] = useFullTiles[i];
    useFullTileBuffers = tmp;
    return *this;
  }
  /// If true all operands unspecified by `useFullTileBuffers` will use the full
  /// view, otherwise the partial view.
  bool useFullTileBuffersDefault = false;
  LinalgPromotionOptions &useFullTileBuffersByDefault() {
    useFullTileBuffersDefault = true;
    return *this;
  }
  /// Allow the use of dynamicaly-sized buffers.
  bool dynamicBuffers = false;
  LinalgPromotionOptions &setDynamicBuffers(unsigned dynamic) {
    dynamicBuffers = dynamic;
    return *this;
  }
  /// Alignment of promoted buffer. If `None` do not specify alignment.
  Optional<unsigned> alignment = None;
  LinalgPromotionOptions &setAlignment(unsigned align) {
    alignment = align;
    return *this;
  }
  /// Callback function to do the allocation of the promoted buffer. If None,
  /// then the default allocation scheme of allocating a memref<?xi8> buffer
  /// followed by a view operation is used.
  Optional<AllocBufferCallbackFn> allocationFn = None;
  Optional<DeallocBufferCallbackFn> deallocationFn = None;
  LinalgPromotionOptions &
  setAllocationDeallocationFns(AllocBufferCallbackFn const &allocFn,
                               DeallocBufferCallbackFn const &deallocFn) {
    allocationFn = allocFn;
    deallocationFn = deallocFn;
    return *this;
  }

  /// Callback function to do the copy of data to and from the promoted
  /// subview. If None then a linalg.copy is used.
  Optional<CopyCallbackFn> copyInFn = None;
  Optional<CopyCallbackFn> copyOutFn = None;
  LinalgPromotionOptions &setCopyInOutFns(CopyCallbackFn const &copyIn,
                                          CopyCallbackFn const &copyOut) {
    copyInFn = copyIn;
    copyOutFn = copyOut;
    return *this;
  }
};

/// Promotes the `subViews` into a new buffer allocated at the insertion point
/// `b`. Promotion occurs in 3 steps:
///   1. Create a new buffer for a full tile (i.e. not clipped at the boundary).
///   2. Take a full view on the buffer.
///   3. Take a partial slice of the full view in step 2. and copy into it.
/// Infers statically sized buffers from subViews unless `dynamicBuffers` is
/// true.
///
/// Returns the modified linalg op (the modification happens in place) as well
/// as all the copy ops created.
Optional<LinalgOp> promoteSubViews(OpBuilder &b, LinalgOp op,
                                   LinalgPromotionOptions options,
                                   OperationFolder *folder = nullptr);

/// Emit a suitable vector form for a Linalg op with fully static shape.
void vectorizeLinalgOp(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `LoopTy` with the proper body for `op`.
template <typename LoopTy, typename ConcreteOp>
Optional<LinalgLoops> linalgLowerOpToLoops(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `scf.for` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult linalgOpToLoops(OpBuilder &builder, Operation *op);

/// Emits a loop nest of `scf.parallel` with the proper body for `op`.
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
LogicalResult promoteSubviewsPrecondition(Operation *op,
                                          LinalgPromotionOptions options);

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
  ParallelLoops = 2,
};
using TileSizeComputationFunction =
    std::function<SmallVector<Value, 4>(OpBuilder &, Operation *)>;
struct LinalgTilingOptions {
  /// Computation function that returns the tile sizes for each operation.
  /// Delayed construction of constant tile sizes should occur to interoperate
  /// with folding.
  TileSizeComputationFunction tileSizeComputationFunction = nullptr;
  LinalgTilingOptions &
  setTileSizeComputationFunction(TileSizeComputationFunction &fun) {
    tileSizeComputationFunction = fun;
    return *this;
  }
  /// Set the `tileSizeComputationFunction` to return the values `ts`. The
  /// values must not fold away when tiling. Otherwise, use a more robust
  /// `tileSizeComputationFunction`.
  LinalgTilingOptions &setTileSizes(ValueRange ts) {
    tileSizeComputationFunction = [&](OpBuilder &, Operation *) {
      return SmallVector<Value, 4>(ts.begin(), ts.end());
    };
    return *this;
  }
  /// Convenience function to set the `tileSizeComputationFunction` to a
  /// function that computes tile sizes at the point they are needed. Allows
  /// proper interaction with folding.
  LinalgTilingOptions &setTileSizes(ArrayRef<int64_t> ts);

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

/// Canonicalization patterns relevant to apply after tiling patterns. These are
/// applied automatically by the tiling pass but need to be applied manually
/// when tiling is called programmatically.
OwningRewritePatternList
getLinalgTilingCanonicalizationPatterns(MLIRContext *ctx);

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
/// Apply the `promoteSubViews` transformation as a pattern.
/// `marker` controls LinalgTransformMarker matching and update when specified.
/// See `promoteSubViews` for more details.
struct LinalgBasePromotionPattern : public RewritePattern {
  LinalgBasePromotionPattern(StringRef opName, MLIRContext *context,
                             LinalgPromotionOptions options,
                             LinalgMarker marker = LinalgMarker(),
                             PatternBenefit benefit = 1);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgMarker marker;
  /// Promotion options.
  LinalgPromotionOptions options;
};

template <typename OpTy>
struct LinalgPromotionPattern : public LinalgBasePromotionPattern {
  LinalgPromotionPattern(MLIRContext *context, LinalgPromotionOptions options,
                         LinalgMarker marker = LinalgMarker(),
                         PatternBenefit benefit = 1)
      : LinalgBasePromotionPattern(OpTy::getOperationName(), context, options,
                                   marker, benefit) {}
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
  /// Controls whether the pattern lowers to library calls, scf.for, affine.for
  /// or scf.parallel.
  LinalgLoweringType loweringType;
};

//===----------------------------------------------------------------------===//
// Support for staged pattern application.
//===----------------------------------------------------------------------===//
/// Helper function to allow applying rewrite patterns, interleaved with more
/// global transformations, in a staged fashion:
///   1. the first stage consists of a list of OwningRewritePatternList. Each
///   OwningRewritePatternList in this list is applied once, in order.
///   2. the second stage consists of a single OwningRewritePattern that is
///   applied greedily until convergence.
///   3. the third stage consists of applying a lambda, generally used for
///   non-local transformation effects. This allows creating custom fused
///   transformations where patterns can be ordered and applied at a finer
///   granularity than a sequence of traditional compiler passes.
LogicalResult applyStagedPatterns(
    Operation *op, ArrayRef<OwningRewritePatternList> stage1Patterns,
    const OwningRewritePatternList &stage2Patterns,
    llvm::function_ref<LogicalResult(Operation *)> stage3Lambda = nullptr);
} // namespace linalg
} // namespace mlir

#endif // DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H_
