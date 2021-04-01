//===- VectorTransforms.h - Vector transformations as patterns --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_VECTOR_VECTORTRANSFORMS_H_
#define DIALECT_VECTOR_VECTORTRANSFORMS_H_

#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class VectorTransferOpInterface;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

namespace scf {
class IfOp;
} // namespace scf

/// Collect a set of patterns to convert from the Vector dialect to itself.
/// Should be merged with populateVectorToSCFLoweringPattern.
void populateVectorToVectorConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns,
    ArrayRef<int64_t> coarseVectorShape = {},
    ArrayRef<int64_t> fineVectorShape = {});

namespace vector {

/// Entry point for unrolling declarative pattern rewrites.
/// `op` is unrolled to the `targetShape` as follows, for each of its operands:
///   1. the unrolled type `unrolledVectorType` and number of unrolled instances
///   `numUnrolledInstances` are computed from the `targetShape`. For now it is
///   assumed the unrolling factors divide the vector sizes.
///   2. a fakeFork cast op is inserted that takes the operand and returns
///   `numUnrolledInstances` results of type `unrolledVectorType`.
///   3. the original op is cloned `numUnrolledInstances` times, once for each
///   result of the fakeFork cast op.
///   4. a fakeJoin cast op takes all these results and merges them into a
///   single aggregate vector result whose size matches the original
///   non-unrolled op operand types.
///
/// Example:
///
///    opA(operand0, operand1)  // numUnrolledInstances = 3
///
///            operand0                   operand1
///               |                          |
///             fork                       fork
///        <----------gather all fork ops --------->
///              /|\                        /|\
///          f00 f01 f02                f10 f11 f12
///        <---------- clone op 3 times --------->
///          opA0(f00, f10), opA1(f01, f11), opA2(f02, f12)
///                 \            |            /
///      <-------------------- join ------------------------->
///
/// Other local patterns then kick in iteratively (including DCE) and compose
/// until all the fakeFork and fakeJoin ops are removed.
///
/// This will be extended in the future to support more advanced use cases than
/// simple pointwise ops.
SmallVector<Value, 1> unrollSingleResultVectorOp(OpBuilder &builder,
                                                 Operation *op,
                                                 ArrayRef<int64_t> targetShape);

/// Unroll a transfer_write op. Break up the vector source into a tuple of
/// vectors matching the given shape. Then store each element with its own
/// transfer_write. If the transfer_write takes a tensor source, return the
/// unrolled Value in result.
///
/// Example:
/// vector.transfer_write %A, %M[%c0, %c0] : vector<4x4xf32>, memref<4x4xf32>
/// ->
/// %0 = vector.extract_slices %A, [2, 4], [1, 1] :
///                vector<4x4xf32> into tuple<vector<2x4xf32>, vector<2x4xf32>>
/// %1 = vector.tuple_get %0, 0 : tuple<vector<2x4xf32>, vector<2x4xf32>>
/// vector.transfer_write %1, %M[%c0, %c0] : vector<2x4xf32>, memref<4x4xf32>
/// %2 = vector.tuple_get %0, 1 : tuple<vector<2x4xf32>, vector<2x4xf32>>
/// vector.transfer_write %2, %M[%c2, %c0] : vector<2x4xf32>, memref<4x4xf32>
LogicalResult unrollTransferWriteOp(OpBuilder &builder, Operation *op,
                                    ArrayRef<int64_t> targetShape,
                                    SmallVector<Value, 1> &result);

/// Options that control the vector unrolling.
struct UnrollVectorOptions {
  using FilterConstraintFnType = std::function<LogicalResult(Operation *op)>;
  /// Callback function that indicates whether vector unrolling should be
  /// attempted on the operation.
  FilterConstraintFnType filterConstraint = nullptr;
  UnrollVectorOptions &setFilterConstraint(FilterConstraintFnType constraint) {
    filterConstraint = constraint;
    return *this;
  }

  using NativeShapeFnType =
      std::function<Optional<SmallVector<int64_t, 4>>(Operation *op)>;
  /// Function that returns the shape of the vector to unroll to for a given
  /// operation. The unrolling is aborted if the function returns `llvm::None`.
  NativeShapeFnType nativeShape = nullptr;
  UnrollVectorOptions &setNativeShapeFn(NativeShapeFnType fn) {
    nativeShape = fn;
    return *this;
  }

  /// Set the native shape to use for unrolling.
  UnrollVectorOptions &setNativeShape(ArrayRef<int64_t> shape) {
    SmallVector<int64_t, 4> tsShape(shape.begin(), shape.end());
    nativeShape = [=](Operation *) -> Optional<SmallVector<int64_t, 4>> {
      return tsShape;
    };
    return *this;
  }
};
/// Pattern to apply `unrollSingleResultVectorOp` to a `targetShape`
/// declaratively.
struct UnrollVectorPattern : public RewritePattern {
  using FilterConstraintType = std::function<LogicalResult(Operation *op)>;
  UnrollVectorPattern(MLIRContext *context, UnrollVectorOptions options)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        options(options) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (options.filterConstraint && failed(options.filterConstraint(op)))
      return failure();
    if (!options.nativeShape) {
      return op->emitError("vector unrolling expects the native shape or native"
                           "shape call back function to be set");
    }
    auto unrollableVectorOp = dyn_cast<VectorUnrollOpInterface>(op);
    if (!unrollableVectorOp)
      return failure();
    auto maybeUnrollShape = unrollableVectorOp.getShapeForUnroll();
    if (!maybeUnrollShape)
      return failure();
    Optional<SmallVector<int64_t, 4>> targetShape = options.nativeShape(op);
    if (!targetShape)
      return op->emitError("failed to get target shape for vector unroll");
    auto maybeShapeRatio = shapeRatio(*maybeUnrollShape, *targetShape);
    if (!maybeShapeRatio ||
        llvm::all_of(*maybeShapeRatio, [](int64_t v) { return v == 1; }))
      return failure();
    if (isa<TransferWriteOp>(op)) {
      SmallVector<Value, 1> result;
      if (failed(unrollTransferWriteOp(rewriter, op, *targetShape, result)))
        return failure();
      rewriter.replaceOp(op, result);
      return success();
    }
    if (op->getNumResults() != 1)
      return failure();
    auto resultVector = unrollSingleResultVectorOp(rewriter, op, *targetShape);
    if (resultVector.size() != 1)
      return failure();
    rewriter.replaceOp(op, resultVector.front());
    return success();
  }

private:
  UnrollVectorOptions options;
};

/// Split a vector.transfer operation into an in-bounds (i.e., no out-of-bounds
/// masking) fastpath and a slowpath.
/// If `ifOp` is not null and the result is `success, the `ifOp` points to the
/// newly created conditional upon function return.
/// To accomodate for the fact that the original vector.transfer indexing may be
/// arbitrary and the slow path indexes @[0...0] in the temporary buffer, the
/// scf.if op returns a view and values of type index.
/// At this time, only vector.transfer_read case is implemented.
///
/// Example (a 2-D vector.transfer_read):
/// ```
///    %1 = vector.transfer_read %0[...], %pad : memref<A...>, vector<...>
/// ```
/// is transformed into:
/// ```
///    %1:3 = scf.if (%inBounds) {
///      // fastpath, direct cast
///      memref.cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view : compatibleMemRefType, index, index
///    } else {
///      // slowpath, not in-bounds vector.transfer or linalg.copy.
///      memref.cast %alloc: memref<B...> to compatibleMemRefType
///      scf.yield %4 : compatibleMemRefType, index, index
//     }
///    %0 = vector.transfer_read %1#0[%1#1, %1#2] {in_bounds = [true ... true]}
/// ```
/// where `alloc` is a top of the function alloca'ed buffer of one vector.
///
/// Preconditions:
///  1. `xferOp.permutation_map()` must be a minor identity map
///  2. the rank of the `xferOp.memref()` and the rank of the `xferOp.vector()`
///  must be equal. This will be relaxed in the future but requires
///  rank-reducing subviews.
LogicalResult
splitFullAndPartialTransferPrecondition(VectorTransferOpInterface xferOp);
LogicalResult splitFullAndPartialTransfer(
    OpBuilder &b, VectorTransferOpInterface xferOp,
    VectorTransformsOptions options = VectorTransformsOptions(),
    scf::IfOp *ifOp = nullptr);

/// Apply `splitFullAndPartialTransfer` selectively via a pattern. This pattern
/// may take an extra filter to perform selection at a finer granularity.
struct VectorTransferFullPartialRewriter : public RewritePattern {
  using FilterConstraintType =
      std::function<LogicalResult(VectorTransferOpInterface op)>;

  explicit VectorTransferFullPartialRewriter(
      MLIRContext *context,
      VectorTransformsOptions options = VectorTransformsOptions(),
      FilterConstraintType filter =
          [](VectorTransferOpInterface op) { return success(); },
      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context), options(options),
        filter(filter) {}

  /// Performs the rewrite.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  VectorTransformsOptions options;
  FilterConstraintType filter;
};

struct DistributeOps {
  ExtractMapOp extract;
  InsertMapOp insert;
};

/// Distribute a N-D vector pointwise operation over a range of given ids taking
/// *all* values in [0 .. multiplicity - 1] (e.g. loop induction variable or
/// SPMD id). This transformation only inserts
/// vector.extract_map/vector.insert_map. It is meant to be used with
/// canonicalizations pattern to propagate and fold the vector
/// insert_map/extract_map operations.
/// Transforms:
//  %v = addf %a, %b : vector<32xf32>
/// to:
/// %v = addf %a, %b : vector<32xf32>
/// %ev = vector.extract_map %v, %id, 32 : vector<32xf32> into vector<1xf32>
/// %nv = vector.insert_map %ev, %id, 32 : vector<1xf32> into vector<32xf32>
Optional<DistributeOps>
distributPointwiseVectorOp(OpBuilder &builder, Operation *op,
                           ArrayRef<Value> id, ArrayRef<int64_t> multiplicity,
                           const AffineMap &map);
/// Canonicalize an extra element using the result of a pointwise operation.
/// Transforms:
/// %v = addf %a, %b : vector32xf32>
/// %dv = vector.extract_map %v, %id, 32 : vector<32xf32> into vector<1xf32>
/// to:
/// %da = vector.extract_map %a, %id, 32 : vector<32xf32> into vector<1xf32>
/// %db = vector.extract_map %a, %id, 32 : vector<32xf32> into vector<1xf32>
/// %dv = addf %da, %db : vector<1xf32>
struct PointwiseExtractPattern : public OpRewritePattern<ExtractMapOp> {
  using FilterConstraintType = std::function<LogicalResult(ExtractMapOp op)>;
  PointwiseExtractPattern(
      MLIRContext *context, FilterConstraintType constraint =
                                [](ExtractMapOp op) { return success(); })
      : OpRewritePattern<ExtractMapOp>(context), filter(constraint) {}
  LogicalResult matchAndRewrite(ExtractMapOp extract,
                                PatternRewriter &rewriter) const override;

private:
  FilterConstraintType filter;
};

/// Implements transfer op write to read forwarding and dead transfer write
/// optimizations.
void transferOpflowOpt(FuncOp func);

} // namespace vector

//===----------------------------------------------------------------------===//
// Finer-grained patterns exposed for more control over individual lowerings.
//===----------------------------------------------------------------------===//

/// Progressive lowering of a `vector.contract %a, %b, %c` with row-major matmul
/// semantics to:
/// ```
///    %flattened_a = vector.shape_cast %a
///    %flattened_b = vector.shape_cast %b
///    %flattened_d = vector.matmul %flattened_a, %flattened_b
///    %d = vector.shape_cast %%flattened_d
///    %e = add %c, %d
/// ```
/// `vector.matmul` later lowers to `llvm.matrix.multiply`.
//
/// This only kicks in when VectorTransformsOptions is set to OuterProduct and
/// the vector.contract op is a row-major matrix multiply.
class ContractionOpToMatmulOpLowering
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;
  using FilterConstraintType =
      std::function<LogicalResult(vector::ContractionOp op)>;

  static LogicalResult defaultFilter(vector::ContractionOp op) {
    return success();
  }

  ContractionOpToMatmulOpLowering(
      vector::VectorTransformsOptions vectorTransformsOptions,
      MLIRContext *context, FilterConstraintType constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context),
        vectorTransformsOptions(vectorTransformsOptions), filter(constraint) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformsOptions;
  FilterConstraintType filter;
};

/// Progressive lowering of a `vector.contract %a, %b, %c` with row-major matmul
/// semantics to a reduction_size-unrolled sequence:
/// ```
///    %at = vector.transpose %a, [1, 0]
///    %bRow0 = vector.extract %b[0]
///    %atRow0 = vector.extract %at[0]
///    %c0 = vector.outerproduct %atRow0, %bRow0, %c
///    ...
///    %bRowK = vector.extract %b[K]
///    %atRowK = vector.extract %at[K]
///    %cK = vector.outerproduct %atRowK, %bRowK, %cK-1
/// ```
///
/// This only kicks in when VectorTransformsOptions is set to OuterProduct and
/// the vector.contract op is a row-major matrix multiply.
class ContractionOpToOuterProductOpLowering
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;
  using FilterConstraintType =
      std::function<LogicalResult(vector::ContractionOp op)>;

  static LogicalResult defaultFilter(vector::ContractionOp op) {
    return success();
  }

  ContractionOpToOuterProductOpLowering(
      vector::VectorTransformsOptions vectorTransformsOptions,
      MLIRContext *context, FilterConstraintType constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context),
        vectorTransformsOptions(vectorTransformsOptions), filter(constraint) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformsOptions;
  FilterConstraintType filter;
};

/// Progressive lowering of a `vector.contract %a, %b, %c` with row-major matmul
/// semantics to an output-size-unrolled sequence:
/// ```
///    %out = constant ... : vector<MxNxelt_type>
///    %bt = vector.transpose %b, [1, 0]
///    %aRow0 = vector.extract %a[0]
///    %btRow0 = vector.extract %bt[0]
///    %c00 = vector.reduce %atRow0, %bRow0
///    %out00 = vector.insert %c00, %out[0, 0]
///    ...
///    %aRowLast = vector.extract %at[M-1]
///    %btRowLast = vector.extract %b[N-1]
///    %cLastLast = vector.reduce %atRowLast, %bRowLast
///    %outcLastLast = vector.insert %cLastLast, %out[M-1, N-1]
/// ```
///
/// This only kicks in when VectorTransformsOptions is set to Dot and
/// the vector.contract op is a row-major matmul or matvec.
class ContractionOpToDotLowering
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;
  using FilterConstraintType =
      std::function<LogicalResult(vector::ContractionOp op)>;

  static LogicalResult defaultFilter(vector::ContractionOp op) {
    return success();
  }

  ContractionOpToDotLowering(
      vector::VectorTransformsOptions vectorTransformsOptions,
      MLIRContext *context, FilterConstraintType constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context),
        vectorTransformsOptions(vectorTransformsOptions),
        filter(defaultFilter) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformsOptions;
  FilterConstraintType filter;
};

/// Progressive lowering of ContractionOp.
///
/// One:
///   %x = vector.contract with at least one free/batch dimension
/// is replaced by:
///   %a = vector.contract with one less free/batch dimension
///   %b = vector.contract with one less free/batch dimension
///   ..
///   %x = combine %a %b ..
/// until a pure contraction is reached (no free/batch dimensions),
/// which is replaced by a dot-product.
///
/// This only kicks in when either VectorTransformsOptions is set
/// to Dot or when other contraction patterns fail.
class ContractionOpLowering : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;
  using FilterConstraintType =
      std::function<LogicalResult(vector::ContractionOp op)>;

  static LogicalResult defaultFilter(vector::ContractionOp op) {
    return success();
  }

  ContractionOpLowering(vector::VectorTransformsOptions vectorTransformsOptions,
                        MLIRContext *context,
                        FilterConstraintType constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context),
        vectorTransformsOptions(vectorTransformsOptions), filter(constraint) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformsOptions;
  FilterConstraintType filter;
  // Lower one parallel dimension.
  Value lowerParallel(vector::ContractionOp op, int64_t lhsIndex,
                      int64_t rhsIndex, PatternRewriter &rewriter) const;
  // Lower one reduction dimension.
  Value lowerReduction(vector::ContractionOp op,
                       PatternRewriter &rewriter) const;
};

} // namespace mlir

#endif // DIALECT_VECTOR_VECTORTRANSFORMS_H_
