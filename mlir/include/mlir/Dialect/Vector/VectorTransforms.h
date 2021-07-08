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

/// Collect a set of pattern to unroll vector operations to a smaller shapes.
/// `options` structure controls which operations are unrolled and the target
/// shape.
/// `op` is unrolled to the `targetShape` as follows, for each of its operands:
///   1. the unrolled type `unrolledVectorType` and number of unrolled instances
///   `numUnrolledInstances` are computed from the `targetShape`. For now it is
///   assumed the unrolling factors divide the vector sizes.
///   2. ExtractStridedSlice are created to break-up the vector operands.
///   3. the original op is cloned `numUnrolledInstances` times, once for each
///   result.
///   4. InsertStridedSlice are inserted to re-assemble the slices into the
///   original vectore shape.
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
/// to combine the ExtractStridedSlice/InsertStridedSlice.
void populateVectorUnrollPatterns(RewritePatternSet &patterns,
                                  const UnrollVectorOptions &options);

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
