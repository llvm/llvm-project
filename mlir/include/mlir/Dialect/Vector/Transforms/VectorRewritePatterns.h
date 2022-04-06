//===- VectorRewritePatterns.h - Vector rewrite patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORREWRITEPATTERNS_H
#define MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORREWRITEPATTERNS_H

#include <utility>

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class RewritePatternSet;

namespace vector {

//===----------------------------------------------------------------------===//
// Vector transformation options exposed as auxiliary structs.
//===----------------------------------------------------------------------===//
/// Enum to control the lowering of `vector.transpose` operations.
enum class VectorTransposeLowering {
  /// Lower transpose into element-wise extract and inserts.
  EltWise = 0,
  /// Lower 2-D transpose to `vector.flat_transpose`, maps 1-1 to LLVM matrix
  /// intrinsics.
  Flat = 1,
  /// Lower 2-D transpose to `vector.shuffle`.
  Shuffle = 2,
};
/// Enum to control the lowering of `vector.multi_reduction` operations.
enum class VectorMultiReductionLowering {
  /// Lower multi_reduction into outer-reduction and inner-parallel ops.
  InnerParallel = 0,
  /// Lower multi_reduction into outer-parallel and inner-reduction ops.
  InnerReduction = 1,
};
/// Enum to control the lowering of `vector.contract` operations.
enum class VectorContractLowering {
  /// Progressively lower to finer grained `vector.contract` and dot-products.
  Dot = 0,
  /// Lower to `vector.matrix_multiply`, maps 1-1 to LLVM matrix intrinsics.
  Matmul = 1,
  /// Lower to `vector.outerproduct`.
  OuterProduct = 2,
};
/// Enum to control the splitting of `vector.transfer` operations into
/// in-bounds and out-of-bounds variants.
enum class VectorTransferSplit {
  /// Do not split vector transfer operations.
  None = 0,
  /// Split using in-bounds + out-of-bounds vector.transfer operations.
  VectorTransfer = 1,
  /// Split using an in-bounds vector.transfer + linalg.fill + linalg.copy
  /// operations.
  LinalgCopy = 2,
  /// Do not split vector transfer operation but instead mark it as "in-bounds".
  ForceInBounds = 3
};
/// Structure to control the behavior of vector transform patterns.
struct VectorTransformsOptions {
  /// Option to control the lowering of vector.contract.
  VectorContractLowering vectorContractLowering = VectorContractLowering::Dot;
  VectorTransformsOptions &
  setVectorTransformsOptions(VectorContractLowering opt) {
    vectorContractLowering = opt;
    return *this;
  }
  /// Option to control the lowering of vector.multi_reduction.
  VectorMultiReductionLowering vectorMultiReductionLowering =
      VectorMultiReductionLowering::InnerParallel;
  VectorTransformsOptions &
  setVectorMultiReductionLowering(VectorMultiReductionLowering opt) {
    vectorMultiReductionLowering = opt;
    return *this;
  }
  /// Option to control the lowering of vector.transpose.
  VectorTransposeLowering vectorTransposeLowering =
      VectorTransposeLowering::EltWise;
  VectorTransformsOptions &
  setVectorTransposeLowering(VectorTransposeLowering opt) {
    vectorTransposeLowering = opt;
    return *this;
  }
  /// Option to control the splitting of vector transfers.
  VectorTransferSplit vectorTransferSplit = VectorTransferSplit::None;
  VectorTransformsOptions &setVectorTransferSplit(VectorTransferSplit opt) {
    vectorTransferSplit = opt;
    return *this;
  }
};

/// Options that control the vector unrolling.
struct UnrollVectorOptions {
  using FilterConstraintFnType = std::function<LogicalResult(Operation *op)>;
  /// Callback function that indicates whether vector unrolling should be
  /// attempted on the operation.
  FilterConstraintFnType filterConstraint = nullptr;
  UnrollVectorOptions &setFilterConstraint(FilterConstraintFnType constraint) {
    filterConstraint = std::move(constraint);
    return *this;
  }

  using NativeShapeFnType =
      std::function<Optional<SmallVector<int64_t, 4>>(Operation *op)>;
  /// Function that returns the shape of the vector to unroll to for a given
  /// operation. The unrolling is aborted if the function returns `llvm::None`.
  NativeShapeFnType nativeShape = nullptr;
  UnrollVectorOptions &setNativeShapeFn(NativeShapeFnType fn) {
    nativeShape = std::move(fn);
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

//===----------------------------------------------------------------------===//
// Vector transformation exposed as populate functions over rewrite patterns.
//===----------------------------------------------------------------------===//

/// Insert TransposeLowering patterns into extraction/insertion.
void populateVectorTransposeLoweringPatterns(
    RewritePatternSet &patterns,
    VectorTransformsOptions options = VectorTransformsOptions());

/// Collect a set of patterns to convert vector.multi_reduction op into
/// a sequence of vector.reduction ops. The patterns comprise:
/// - InnerOuterDimReductionConversion: rewrites vector.multi_reduction such
/// that all reduction dimensions are either innermost or outermost, by adding
/// the proper vector.transpose operations.
/// - ReduceMultiDimReductionRank: once in innermost or outermost reduction
/// form, rewrites n-D vector.multi_reduction into 2-D vector.multi_reduction,
/// by introducing vector.shape_cast ops to collapse + multi-reduce + expand
/// back.
/// - TwoDimMultiReductionToElementWise: once in 2-D vector.multi_reduction
/// form, with an **outermost** reduction dimension, unroll the outer dimension
/// to obtain a sequence of 1-D vector ops. This also has an opportunity for
/// tree-reduction (in the future).
/// - TwoDimMultiReductionToReduction: once in 2-D vector.multi_reduction form,
/// with an **innermost** reduction dimension, unroll the outer dimension to
/// obtain a sequence of extract + vector.reduction + insert. This can further
/// lower to horizontal reduction ops.
/// - OneDimMultiReductionToTwoDim: for cases that reduce to 1-D vector<k>
/// reduction (and are thus missing either a parallel or a reduction), we lift
/// them back up to 2-D with a simple vector.shape_cast to vector<1xk> so that
/// the other patterns can kick in, thus fully exiting out of the
/// vector.multi_reduction abstraction.
void populateVectorMultiReductionLoweringPatterns(
    RewritePatternSet &patterns, VectorMultiReductionLowering options);

/// Collects patterns to progressively lower vector contraction ops on high-D
/// into low-D reduction and product ops.
void populateVectorContractLoweringPatterns(
    RewritePatternSet &patterns,
    VectorTransformsOptions options = VectorTransformsOptions());

/// Collect patterns to convert reduction op to vector.contract and fold
/// transpose/broadcast ops into the contract.
void populateVectorReductionToContractPatterns(RewritePatternSet &patterns);

/// Collect patterns to convert scan op
void populateVectorScanLoweringPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Vector.transfer patterns.
//===----------------------------------------------------------------------===//
/// Collect a set of transfer read/write lowering patterns that simplify the
/// permutation map (e.g., converting it to a minor identity map) by inserting
/// broadcasts and transposes. More specifically:
///
/// [TransferReadPermutationLowering]
/// Lower transfer_read op with permutation into a transfer_read with a
/// permutation map composed of leading zeros followed by a minor identity +
/// vector.transpose op.
/// Ex:
///     vector.transfer_read ...
///         permutation_map: (d0, d1, d2) -> (0, d1)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2) -> (d1, 0)
///     vector.transpose %v, [1, 0]
///
///     vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, 0, 0, d1, d3)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, 0, d1, 0, d3)
///     vector.transpose %v, [0, 1, 3, 2, 4]
/// Note that an alternative is to transform it to linalg.transpose +
/// vector.transfer_read to do the transpose in memory instead.
///
/// [TransferWritePermutationLowering]
/// Lower transfer_write op with permutation into a transfer_write with a
/// minor identity permutation map. (transfer_write ops cannot have broadcasts.)
/// Ex:
///     vector.transfer_write %v ...
///         permutation_map: (d0, d1, d2) -> (d2, d0, d1)
/// into:
///     %tmp = vector.transpose %v, [2, 0, 1]
///     vector.transfer_write %tmp ...
///         permutation_map: (d0, d1, d2) -> (d0, d1, d2)
///
///     vector.transfer_write %v ...
///         permutation_map: (d0, d1, d2, d3) -> (d3, d2)
/// into:
///     %tmp = vector.transpose %v, [1, 0]
///     %v = vector.transfer_write %tmp ...
///         permutation_map: (d0, d1, d2, d3) -> (d2, d3)
///
/// [TransferOpReduceRank]
/// Lower transfer_read op with broadcast in the leading dimensions into
/// transfer_read of lower rank + vector.broadcast.
/// Ex: vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, d1, 0, d3)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (d1, 0, d3)
///     vector.broadcast %v
void populateVectorTransferPermutationMapLoweringPatterns(
    RewritePatternSet &patterns);

/// Collect a set of patterns to reduce the rank of the operands of vector
/// transfer ops to operate on the largest contigious vector.
/// These patterns are useful when lowering to dialects with 1d vector type
/// such as llvm and it will result fewer memory reads.
void populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
    RewritePatternSet &patterns);

/// Populate `patterns` with the following patterns.
///
/// [DecomposeDifferentRankInsertStridedSlice]
/// ==========================================
/// RewritePattern for InsertStridedSliceOp where source and destination vectors
/// have different ranks.
///
/// When ranks are different, InsertStridedSlice needs to extract a properly
/// ranked vector from the destination vector into which to insert. This pattern
/// only takes care of this extraction part and forwards the rest to
/// [VectorInsertStridedSliceOpSameRankRewritePattern].
///
/// For a k-D source and n-D destination vector (k < n), we emit:
///   1. ExtractOp to extract the (unique) (n-1)-D subvector into which to
///      insert the k-D source.
///   2. k-D -> (n-1)-D InsertStridedSlice op
///   3. InsertOp that is the reverse of 1.
///
/// [DecomposeNDExtractStridedSlice]
/// ================================
/// For such cases, we can rewrite it to ExtractOp/ExtractElementOp + lower
/// rank ExtractStridedSliceOp + InsertOp/InsertElementOp for the n-D case.
void populateVectorInsertExtractStridedSliceDecompositionPatterns(
    RewritePatternSet &patterns);

/// Populate `patterns` with the following patterns.
///
/// Patterns in populateVectorInsertExtractStridedSliceDecompositionPatterns();
///
/// [ConvertSameRankInsertStridedSliceIntoShuffle]
/// ==============================================
/// RewritePattern for InsertStridedSliceOp where source and destination vectors
/// have the same rank. For each outermost index in the slice:
///   begin    end             stride
/// [offset : offset+size*stride : stride]
///   1. ExtractOp one (k-1)-D source subvector and one (n-1)-D dest subvector.
///   2. InsertStridedSlice (k-1)-D into (n-1)-D
///   3. the destination subvector is inserted back in the proper place
///   3. InsertOp that is the reverse of 1.
///
/// [Convert1DExtractStridedSliceIntoShuffle]
/// =========================================
/// For such cases, we can lower it to a ShuffleOp.
void populateVectorInsertExtractStridedSliceTransforms(
    RewritePatternSet &patterns);

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

//===----------------------------------------------------------------------===//
// Finer-grained patterns exposed for more control over individual lowerings.
//===----------------------------------------------------------------------===//
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
        filter(std::move(filter)) {}

  /// Performs the rewrite.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  VectorTransformsOptions options;
  FilterConstraintType filter;
};

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
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context, FilterConstraintType constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context),
        vectorTransformOptions(vectorTransformOptions),
        filter(std::move(constraint)) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
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
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context, FilterConstraintType constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context),
        vectorTransformOptions(vectorTransformOptions),
        filter(std::move(constraint)) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
  FilterConstraintType filter;
};

/// Progressive lowering of a `vector.contract %a, %b, %c` with row-major matmul
/// semantics to an output-size-unrolled sequence:
/// ```
///    %out = arith.constant ... : vector<MxNxelt_type>
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
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context,
      const FilterConstraintType &constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context),
        vectorTransformOptions(vectorTransformOptions), filter(defaultFilter) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
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

  ContractionOpLowering(vector::VectorTransformsOptions vectorTransformOptions,
                        MLIRContext *context,
                        FilterConstraintType constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context),
        vectorTransformOptions(vectorTransformOptions),
        filter(std::move(constraint)) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
  FilterConstraintType filter;
  // Lower one parallel dimension.
  Value lowerParallel(vector::ContractionOp op, int64_t lhsIndex,
                      int64_t rhsIndex, PatternRewriter &rewriter) const;
  // Lower one reduction dimension.
  Value lowerReduction(vector::ContractionOp op,
                       PatternRewriter &rewriter) const;
};

} // namespace vector
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORREWRITEPATTERNS_H
