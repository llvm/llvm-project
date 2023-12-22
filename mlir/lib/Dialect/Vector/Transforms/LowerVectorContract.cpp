//===- LowerVectorContract.cpp - Lower 'vector.contract' operation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.contract' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "vector-contract-lowering"

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

// Helper to find an index in an affine map.
static std::optional<int64_t> getResultIndex(AffineMap map, int64_t index) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
    int64_t idx = map.getDimPosition(i);
    if (idx == index)
      return i;
  }
  return std::nullopt;
}

// Helper to construct iterator types with one index removed.
static SmallVector<Attribute> adjustIter(ArrayAttr iteratorTypes,
                                         int64_t index) {
  SmallVector<Attribute> results;
  for (const auto &it : llvm::enumerate(iteratorTypes)) {
    int64_t idx = it.index();
    if (idx == index)
      continue;
    results.push_back(it.value());
  }
  return results;
}

// Helper to construct an affine map with one index removed.
static AffineMap adjustMap(AffineMap map, int64_t index,
                           PatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  SmallVector<AffineExpr> results;
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
    int64_t idx = map.getDimPosition(i);
    if (idx == index)
      continue;
    // Re-insert remaining indices, but renamed when occurring
    // after the removed index.
    auto targetExpr = getAffineDimExpr(idx < index ? idx : idx - 1, ctx);
    results.push_back(targetExpr);
  }
  return AffineMap::get(map.getNumDims() - 1, 0, results, ctx);
}

// Helper method to possibly drop a dimension in a load.
// TODO
static Value reshapeLoad(Location loc, Value val, VectorType type,
                         int64_t index, int64_t pos,
                         PatternRewriter &rewriter) {
  if (index == -1)
    return val;

  // At extraction dimension?
  if (index == 0)
    return rewriter.create<vector::ExtractOp>(loc, val, pos);

  // Unroll leading dimensions.
  VectorType vType = VectorType::Builder(type).dropDim(0);
  VectorType resType = VectorType::Builder(type).dropDim(index);
  Value result = rewriter.create<arith::ConstantOp>(
      loc, resType, rewriter.getZeroAttr(resType));
  for (int64_t d = 0, e = resType.getDimSize(0); d < e; d++) {
    Value ext = rewriter.create<vector::ExtractOp>(loc, val, d);
    Value load = reshapeLoad(loc, ext, vType, index - 1, pos, rewriter);
    result = rewriter.create<vector::InsertOp>(loc, load, result, d);
  }
  return result;
}

// Helper method to possibly drop a dimension in a store.
// TODO
static Value reshapeStore(Location loc, Value val, Value result,
                          VectorType type, int64_t index, int64_t pos,
                          PatternRewriter &rewriter) {
  // Unmodified?
  if (index == -1)
    return val;
  // At insertion dimension?
  if (index == 0)
    return rewriter.create<vector::InsertOp>(loc, val, result, pos);

  // Unroll leading dimensions.
  VectorType vType = VectorType::Builder(type).dropDim(0);
  for (int64_t d = 0, e = type.getDimSize(0); d < e; d++) {
    Value ext = rewriter.create<vector::ExtractOp>(loc, result, d);
    Value ins = rewriter.create<vector::ExtractOp>(loc, val, d);
    Value sto = reshapeStore(loc, ins, ext, vType, index - 1, pos, rewriter);
    result = rewriter.create<vector::InsertOp>(loc, sto, result, d);
  }
  return result;
}

/// Helper to create arithmetic operation associated with a kind of contraction.
static std::optional<Value>
createContractArithOp(Location loc, Value x, Value y, Value acc,
                      vector::CombiningKind kind, PatternRewriter &rewriter,
                      bool isInt, Value mask = Value()) {
  using vector::CombiningKind;
  Value mul;

  if (isInt) {
    if (kind == CombiningKind::MINNUMF || kind == CombiningKind::MAXNUMF ||
        kind == CombiningKind::MINIMUMF || kind == CombiningKind::MAXIMUMF)
      // Only valid for floating point types.
      return std::nullopt;
    mul = rewriter.create<arith::MulIOp>(loc, x, y);
  } else {
    // Float case.
    if (kind == CombiningKind::AND || kind == CombiningKind::MINUI ||
        kind == CombiningKind::MINSI || kind == CombiningKind::MAXUI ||
        kind == CombiningKind::MAXSI || kind == CombiningKind::OR ||
        kind == CombiningKind::XOR)
      // Only valid for integer types.
      return std::nullopt;
    // Special case for fused multiply-add.
    if (acc && isa<VectorType>(acc.getType()) && kind == CombiningKind::ADD) {
      Value fma = rewriter.create<vector::FMAOp>(loc, x, y, acc);
      if (mask)
        // The fma op doesn't need explicit masking. However, fma ops used in
        // reductions must preserve previous 'acc' values for masked-out lanes.
        fma = selectPassthru(rewriter, mask, fma, acc);
      return fma;
    }
    mul = rewriter.create<arith::MulFOp>(loc, x, y);
  }

  if (!acc)
    return std::optional<Value>(mul);

  return makeArithReduction(rewriter, loc, kind, mul, acc,
                            /*fastmath=*/nullptr, mask);
}

/// Return the positions of the reductions in the given map.
static SmallVector<int64_t> getReductionIndex(AffineMap map,
                                              ArrayAttr iteratorTypes) {
  SmallVector<int64_t> dimsIdx;
  for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
    if (isReductionIterator(iteratorTypes[map.getDimPosition(i)]))
      dimsIdx.push_back(i);
  }
  return dimsIdx;
}

/// Look for a given dimension in an affine map and return its position. Return
/// std::nullopt if the dimension is not in the map results.
static std::optional<unsigned> getDimPosition(AffineMap map, unsigned dim) {
  for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
    if (map.getDimPosition(i) == dim)
      return i;
  }
  return std::nullopt;
}

/// Creates an AddIOp if `isInt` is true otherwise create an arith::AddFOp using
/// operands `x` and `y`.
static Value createAdd(Location loc, Value x, Value y, bool isInt,
                       PatternRewriter &rewriter) {
  if (isInt)
    return rewriter.create<arith::AddIOp>(loc, x, y);
  return rewriter.create<arith::AddFOp>(loc, x, y);
}

/// Creates a MulIOp if `isInt` is true otherwise create an MulFOp using
/// operands `x and `y`.
static Value createMul(Location loc, Value x, Value y, bool isInt,
                       PatternRewriter &rewriter) {
  if (isInt)
    return rewriter.create<arith::MulIOp>(loc, x, y);
  return rewriter.create<arith::MulFOp>(loc, x, y);
}

namespace {

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
  using OpRewritePattern::OpRewritePattern;

  using FilterConstraintType =
      std::function<LogicalResult(vector::ContractionOp op)>;

  static LogicalResult defaultFilter(vector::ContractionOp op) {
    return success();
  }

  ContractionOpToMatmulOpLowering(
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context, PatternBenefit benefit = 1,
      FilterConstraintType constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context, benefit),
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
  using OpRewritePattern::OpRewritePattern;

  using FilterConstraintType =
      std::function<LogicalResult(vector::ContractionOp op)>;

  static LogicalResult defaultFilter(vector::ContractionOp op) {
    return success();
  }

  ContractionOpToOuterProductOpLowering(
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context, PatternBenefit benefit = 1,
      FilterConstraintType constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context, benefit),
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
  using OpRewritePattern::OpRewritePattern;

  using FilterConstraintType =
      std::function<LogicalResult(vector::ContractionOp op)>;

  static LogicalResult defaultFilter(vector::ContractionOp op) {
    return success();
  }

  ContractionOpToDotLowering(
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context, PatternBenefit benefit = 1,
      const FilterConstraintType &constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context, benefit),
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
  using OpRewritePattern::OpRewritePattern;
  using FilterConstraintType =
      std::function<LogicalResult(vector::ContractionOp op)>;

  static LogicalResult defaultFilter(vector::ContractionOp op) {
    return success();
  }

  ContractionOpLowering(vector::VectorTransformsOptions vectorTransformOptions,
                        MLIRContext *context, PatternBenefit benefit = 1,
                        FilterConstraintType constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context, benefit),
        vectorTransformOptions(vectorTransformOptions),
        filter(std::move(constraint)) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
  FilterConstraintType filter;
  // Lower one parallel dimension.
  FailureOr<Value> lowerParallel(PatternRewriter &rewriter,
                                 vector::ContractionOp op, int64_t lhsIndex,
                                 int64_t rhsIndex, Value mask) const;
  // Lower one reduction dimension.
  FailureOr<Value> lowerReduction(PatternRewriter &rewriter,
                                  vector::ContractionOp op, Value mask) const;
};

/// Generate a vector implementation for matmat, matvec and tmatvec.
/// This unrolls outer-products along the reduction dimension.
struct UnrolledOuterProductGenerator
    : public StructuredGenerator<vector::ContractionOp, vector::IteratorType> {
  UnrolledOuterProductGenerator(RewriterBase &b, vector::ContractionOp op)
      : StructuredGenerator<vector::ContractionOp, vector::IteratorType>(b, op),
        kind(op.getKind()), lhs(op.getLhs()), rhs(op.getRhs()),
        res(op.getAcc()), lhsType(op.getLhsType()) {
    auto maskableOp = cast<MaskableOpInterface>(op.getOperation());
    if (maskableOp.isMasked())
      mask = maskableOp.getMaskingOp().getMask();
  }

  Value t(Value v, ArrayRef<int64_t> perm = {1, 0}) {
    if (!v)
      return v;
    return rewriter.create<vector::TransposeOp>(loc, v, perm);
  }

  Value promote(Value v, Type dstElementType) {
    Type elementType = v.getType();
    auto vecType = dyn_cast<VectorType>(elementType);
    if (vecType)
      elementType = vecType.getElementType();
    if (elementType == dstElementType)
      return v;
    Type promotedType = dstElementType;
    if (vecType)
      promotedType = vecType.clone(promotedType);
    if (isa<FloatType>(dstElementType))
      return rewriter.create<arith::ExtFOp>(loc, promotedType, v);
    return rewriter.create<arith::ExtSIOp>(loc, promotedType, v);
  }

  FailureOr<Value> outerProd(Value lhs, Value rhs, Value res,
                             VectorType lhsType, int reductionDim,
                             std::optional<Value> maybeMask = std::nullopt) {
    // Unrolling a scalable dimension would be incorrect - bail out.
    if (lhsType.getScalableDims()[reductionDim])
      return failure();

    int reductionSize = lhsType.getDimSize(reductionDim);
    assert(reductionSize > 0 &&
           "Reduction dim must be a known static size to allow unrolling");

    // Incremental support for masking.
    if (mask && !maybeMask.has_value())
      return failure();

    Type resElementType = cast<VectorType>(res.getType()).getElementType();
    for (int64_t k = 0; k < reductionSize; ++k) {
      Value extractA = rewriter.create<vector::ExtractOp>(loc, lhs, k);
      Value extractB = rewriter.create<vector::ExtractOp>(loc, rhs, k);
      extractA = promote(extractA, resElementType);
      extractB = promote(extractB, resElementType);
      Value extractMask;
      if (maybeMask.has_value() && maybeMask.value())
        extractMask =
            rewriter.create<vector::ExtractOp>(loc, maybeMask.value(), k);

      Operation *outerProdOp = rewriter.create<vector::OuterProductOp>(
          loc, res.getType(), extractA, extractB, res, kind);
      res = maskOperation(rewriter, outerProdOp, extractMask)->getResult(0);
    }
    return res;
  }

  /// Two outer parallel, one inner reduction (matmat flavor).
  FailureOr<Value> matmat() {
    if (!iters({Par(), Par(), Red()}))
      return failure();
    // Set up the parallel/reduction structure in the right form.
    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);
    Value transposedMask = t(mask, {2, 0, 1});
    // Classical row-major matmul:  Just permute the lhs.
    if (layout({{m, k}, {k, n}, {m, n}}))
      return outerProd(t(lhs), rhs, res, lhsType, /*reductionDim=*/1,
                       transposedMask);
    // TODO: may be better to fail and use some vector<k> -> scalar reduction.
    if (layout({{m, k}, {n, k}, {m, n}})) {
      Value tlhs = t(lhs);
      return outerProd(tlhs, t(rhs), res, lhsType, /*reductionDim=*/1,
                       transposedMask);
    }
    // No need to permute anything.
    if (layout({{k, m}, {k, n}, {m, n}}))
      return outerProd(lhs, rhs, res, lhsType, /*reductionDim=*/0,
                       transposedMask);
    // Just permute the rhs.
    if (layout({{k, m}, {n, k}, {m, n}}))
      return outerProd(lhs, t(rhs), res, lhsType, /*reductionDim=*/0,
                       transposedMask);
    // Transposed output: swap RHS and LHS.
    // Classical row-major matmul: permute the lhs.
    if (layout({{m, k}, {k, n}, {n, m}}))
      return outerProd(rhs, t(lhs), res, lhsType, /*reductionDim=*/1,
                       transposedMask);
    // TODO: may be better to fail and use some vector<k> -> scalar reduction.
    if (layout({{m, k}, {n, k}, {n, m}})) {
      Value trhs = t(rhs);
      return outerProd(trhs, t(lhs), res, lhsType, /*reductionDim=*/1,
                       transposedMask);
    }
    if (layout({{k, m}, {k, n}, {n, m}}))
      return outerProd(rhs, lhs, res, lhsType, /*reductionDim=*/0,
                       transposedMask);
    if (layout({{k, m}, {n, k}, {n, m}}))
      return outerProd(t(rhs), lhs, res, lhsType, /*reductionDim=*/0,
                       transposedMask);
    return failure();
  }

  //
  // One outer parallel, one inner reduction (matvec flavor).
  // Mask needs to be transposed everywhere to turn the reduction dimension
  // outermost as required by outerproduct.
  //
  FailureOr<Value> matvec() {
    if (!iters({Par(), Red()}))
      return failure();
    AffineExpr m, k;
    bindDims(rewriter.getContext(), m, k);
    Value transposedMask = t(mask);

    // Case mat-vec: transpose.
    if (layout({{m, k}, {k}, {m}}))
      return outerProd(t(lhs), rhs, res, lhsType, /*reductionDim=*/1,
                       transposedMask);
    // Case mat-trans-vec: ready to go.
    if (layout({{k, m}, {k}, {m}}))
      return outerProd(lhs, rhs, res, lhsType, /*reductionDim=*/0,
                       transposedMask);
    // Case vec-mat: swap and transpose.
    if (layout({{k}, {m, k}, {m}}))
      return outerProd(t(rhs), lhs, res, lhsType, /*reductionDim=*/0,
                       transposedMask);
    // Case vec-mat-trans: swap and ready to go.
    if (layout({{k}, {k, m}, {m}}))
      return outerProd(rhs, lhs, res, lhsType, /*reductionDim=*/0,
                       transposedMask);
    return failure();
  }

  //
  // One outer reduction, one inner parallel (tmatvec flavor).
  // Mask already has the shape of the outer product.
  //
  FailureOr<Value> tmatvec() {
    if (!iters({Red(), Par()}))
      return failure();
    AffineExpr k, m;
    bindDims(rewriter.getContext(), k, m);

    // Case mat-vec: transpose.
    if (layout({{m, k}, {k}, {m}}))
      return outerProd(t(lhs), rhs, res, lhsType, /*reductionDim=*/1, mask);
    // Case mat-trans-vec: ready to go.
    if (layout({{k, m}, {k}, {m}}))
      return outerProd(lhs, rhs, res, lhsType, /*reductionDim=*/0, mask);
    // Case vec-mat: swap and transpose.
    if (layout({{k}, {m, k}, {m}}))
      return outerProd(t(rhs), lhs, res, lhsType, /*reductionDim=*/0, mask);
    // Case vec-mat-trans: swap and ready to go.
    if (layout({{k}, {k, m}, {m}}))
      return outerProd(rhs, lhs, res, lhsType, /*reductionDim=*/0, mask);
    return failure();
  }

private:
  vector::CombiningKind kind;
  Value lhs, rhs, res, mask;
  VectorType lhsType;
};

/// Progressively lower a `vector.contract %a, %b, %c` with row-major matmul
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
/// This only kicks in when VectorTransformsOptions is set to OuterProduct but
/// otherwise supports any layout permutation of the matrix-multiply.
LogicalResult ContractionOpToOuterProductOpLowering::matchAndRewrite(
    vector::ContractionOp op, PatternRewriter &rewriter) const {
  if (vectorTransformOptions.vectorContractLowering !=
      vector::VectorContractLowering::OuterProduct)
    return failure();

  if (failed(filter(op)))
    return failure();

  // Vector mask setup.
  OpBuilder::InsertionGuard guard(rewriter);
  auto maskableOp = cast<vector::MaskableOpInterface>(op.getOperation());
  Operation *rootOp;
  if (maskableOp.isMasked()) {
    rewriter.setInsertionPoint(maskableOp.getMaskingOp());
    rootOp = maskableOp.getMaskingOp();
  } else {
    rootOp = op;
  }

  UnrolledOuterProductGenerator e(rewriter, op);
  FailureOr<Value> matmatRes = e.matmat();
  if (succeeded(matmatRes)) {
    rewriter.replaceOp(rootOp, *matmatRes);
    return success();
  }
  FailureOr<Value> matvecRes = e.matvec();
  if (succeeded(matvecRes)) {
    rewriter.replaceOp(rootOp, *matvecRes);
    return success();
  }
  FailureOr<Value> tmatvecRes = e.tmatvec();
  if (succeeded(tmatvecRes)) {
    rewriter.replaceOp(rootOp, *tmatvecRes);
    return success();
  }

  return failure();
}

LogicalResult
ContractionOpToDotLowering::matchAndRewrite(vector::ContractionOp op,
                                            PatternRewriter &rewriter) const {
  // TODO: Support vector.mask.
  auto maskableOp = cast<MaskableOpInterface>(op.getOperation());
  if (maskableOp.isMasked())
    return failure();

  if (failed(filter(op)))
    return failure();

  if (vectorTransformOptions.vectorContractLowering !=
      vector::VectorContractLowering::Dot)
    return failure();

  auto iteratorTypes = op.getIteratorTypes().getValue();
  static constexpr std::array<int64_t, 2> perm = {1, 0};
  Location loc = op.getLoc();
  Value lhs = op.getLhs(), rhs = op.getRhs();

  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n, k;
  bindDims(rewriter.getContext(), m, n, k);
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  //
  // In the following we wish to make the reduction dimension innermost so we
  // can load vectors and just fmul + reduce into a scalar.
  //
  if (isParallelIterator(iteratorTypes[0]) &&
      isParallelIterator(iteratorTypes[1]) &&
      isReductionIterator(iteratorTypes[2])) {
    //
    // Two outer parallel, one inner reduction (matmat flavor).
    //
    if (maps == infer({{m, k}, {k, n}, {m, n}})) {
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    } else if (maps == infer({{m, k}, {n, k}, {m, n}})) {
      // No need to permute anything.
    } else if (maps == infer({{k, m}, {k, n}, {m, n}})) {
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    } else if (maps == infer({{k, m}, {n, k}, {m, n}})) {
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{m, k}, {k, n}, {n, m}})) {
      // This is the classical row-major matmul. Just permute the lhs.
      Value tmp = lhs;
      lhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
      rhs = tmp;
    } else if (maps == infer({{m, k}, {n, k}, {n, m}})) {
      std::swap(lhs, rhs);
    } else if (maps == infer({{k, m}, {k, n}, {n, m}})) {
      Value tmp = lhs;
      lhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
      rhs = rewriter.create<vector::TransposeOp>(loc, tmp, perm);
    } else if (maps == infer({{k, m}, {n, k}, {n, m}})) {
      Value tmp = rhs;
      rhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
      lhs = tmp;
    } else {
      return failure();
    }
  } else if (isParallelIterator(iteratorTypes[0]) &&
             isReductionIterator(iteratorTypes[1])) {
    //
    // One outer parallel, one inner reduction (matvec flavor)
    //
    if (maps == infer({{m, n}, {n}, {m}})) {
      // No need to permute anything.
    } else if (maps == infer({{n, m}, {n}, {m}})) {
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{n}, {m, n}, {m}})) {
      std::swap(lhs, rhs);
    } else if (maps == infer({{n}, {n, m}, {m}})) {
      std::swap(lhs, rhs);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else {
      return failure();
    }
  } else {
    return failure();
  }

  VectorType dstType = cast<VectorType>(op.getResultType());
  assert(dstType.getRank() >= 1 && dstType.getRank() <= 2 &&
         "Expected dst type of rank 1 or 2");

  unsigned rank = dstType.getRank();
  unsigned dstRows = dstType.getShape()[0];
  unsigned dstColumns = rank == 1 ? 1 : dstType.getShape()[1];

  // ExtractOp does not allow dynamic indexing, we must unroll explicitly.
  Value res = rewriter.create<arith::ConstantOp>(loc, dstType,
                                                 rewriter.getZeroAttr(dstType));
  bool isInt = isa<IntegerType>(dstType.getElementType());
  for (unsigned r = 0; r < dstRows; ++r) {
    Value a = rewriter.create<vector::ExtractOp>(op.getLoc(), lhs, r);
    for (unsigned c = 0; c < dstColumns; ++c) {
      Value b = rank == 1
                    ? rhs
                    : rewriter.create<vector::ExtractOp>(op.getLoc(), rhs, c);
      Value m = createMul(op.getLoc(), a, b, isInt, rewriter);
      Value reduced = rewriter.create<vector::ReductionOp>(
          op.getLoc(), vector::CombiningKind::ADD, m);

      SmallVector<int64_t, 2> pos = rank == 1 ? SmallVector<int64_t, 2>{r}
                                              : SmallVector<int64_t, 2>{r, c};
      res = rewriter.create<vector::InsertOp>(op.getLoc(), reduced, res, pos);
    }
  }
  if (auto acc = op.getAcc())
    res = createAdd(op.getLoc(), res, acc, isInt, rewriter);
  rewriter.replaceOp(op, res);
  return success();
}

/// Lower vector.contract with all size one reduction dimensions to
/// elementwise ops when possible.
struct ContractOpToElementwise
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;
  using FilterConstraintType =
      std::function<LogicalResult(vector::ContractionOp op)>;
  static LogicalResult defaultFilter(vector::ContractionOp op) {
    return success();
  }
  ContractOpToElementwise(
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context, PatternBenefit benefit = 1,
      const FilterConstraintType &constraint = defaultFilter)
      : OpRewritePattern<vector::ContractionOp>(context, benefit),
        vectorTransformOptions(vectorTransformOptions), filter(defaultFilter) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    // TODO: Support vector.mask.
    auto maskableOp = cast<MaskableOpInterface>(contractOp.getOperation());
    if (maskableOp.isMasked())
      return failure();

    if (failed(filter(contractOp)))
      return failure();

    if (vectorTransformOptions.vectorContractLowering !=
        vector::VectorContractLowering::ParallelArith)
      return failure();

    ArrayRef<int64_t> lhsShape = contractOp.getLhsType().getShape();
    ArrayRef<int64_t> rhsShape = contractOp.getRhsType().getShape();
    AffineMap lhsMap = contractOp.getIndexingMapsArray()[0];
    AffineMap rhsMap = contractOp.getIndexingMapsArray()[1];
    SmallVector<int64_t> lhsReductionDims =
        getReductionIndex(lhsMap, contractOp.getIteratorTypes());
    SmallVector<int64_t> rhsReductionDims =
        getReductionIndex(rhsMap, contractOp.getIteratorTypes());
    // All the reduction dimensions must be a size 1.
    for (int64_t dim : lhsReductionDims) {
      if (lhsShape[dim] != 1)
        return failure();
    }
    for (int64_t dim : rhsReductionDims) {
      if (rhsShape[dim] != 1)
        return failure();
    }
    AffineMap accMap = contractOp.getIndexingMapsArray()[2];
    unsigned numParallelDims = accMap.getNumResults();
    unsigned numLhsDimToBroadcast =
        numParallelDims - (lhsMap.getNumResults() - lhsReductionDims.size());
    unsigned numRhsDimToBroadcast =
        numParallelDims - (rhsMap.getNumResults() - rhsReductionDims.size());
    SmallVector<int64_t> lhsDims;
    SmallVector<int64_t> lhsTranspose;
    SmallVector<int64_t> rhsDims;
    SmallVector<int64_t> rhsTranspose;
    for (int64_t dim : lhsReductionDims)
      lhsTranspose.push_back(numLhsDimToBroadcast + dim);
    for (int64_t dim : rhsReductionDims)
      rhsTranspose.push_back(numRhsDimToBroadcast + dim);
    // Loop through the parallel dimensions to calculate the dimensions to
    // broadcast and to permute in order to extract only parallel dimensions.
    for (unsigned i = 0; i < numParallelDims; i++) {
      std::optional<unsigned> lhsDim =
          getDimPosition(lhsMap, accMap.getDimPosition(i));
      if (lhsDim) {
        lhsTranspose.push_back(numLhsDimToBroadcast + *lhsDim);
      } else {
        // If the parallel dimension doesn't exist we will have to broadcast it.
        lhsDims.push_back(
            cast<VectorType>(contractOp.getResultType()).getDimSize(i));
        lhsTranspose.push_back(lhsDims.size() - 1);
      }
      std::optional<unsigned> rhsDim =
          getDimPosition(rhsMap, accMap.getDimPosition(i));
      if (rhsDim) {
        rhsTranspose.push_back(numRhsDimToBroadcast + *rhsDim);
      } else {
        // If the parallel dimension doesn't exist we will have to broadcast it.
        rhsDims.push_back(
            cast<VectorType>(contractOp.getResultType()).getDimSize(i));
        rhsTranspose.push_back(rhsDims.size() - 1);
      }
    }
    Value newLhs = contractOp.getLhs();
    Value newRhs = contractOp.getRhs();
    Location loc = contractOp.getLoc();
    if (!lhsDims.empty()) {
      lhsDims.append(lhsShape.begin(), lhsShape.end());
      auto expandedType =
          VectorType::get(lhsDims, contractOp.getLhsType().getElementType());
      newLhs = rewriter.create<vector::BroadcastOp>(loc, expandedType, newLhs);
    }
    if (!rhsDims.empty()) {
      rhsDims.append(rhsShape.begin(), rhsShape.end());
      auto expandedType =
          VectorType::get(rhsDims, contractOp.getRhsType().getElementType());
      newRhs = rewriter.create<vector::BroadcastOp>(loc, expandedType, newRhs);
    }
    bool isInt = contractOp.getLhsType().getElementType().isIntOrIndex();
    newLhs = rewriter.create<vector::TransposeOp>(loc, newLhs, lhsTranspose);
    newRhs = rewriter.create<vector::TransposeOp>(loc, newRhs, rhsTranspose);
    SmallVector<int64_t> lhsOffsets(lhsReductionDims.size(), 0);
    SmallVector<int64_t> rhsOffsets(rhsReductionDims.size(), 0);
    newLhs = rewriter.create<vector::ExtractOp>(loc, newLhs, lhsOffsets);
    newRhs = rewriter.create<vector::ExtractOp>(loc, newRhs, rhsOffsets);
    std::optional<Value> result =
        createContractArithOp(loc, newLhs, newRhs, contractOp.getAcc(),
                              contractOp.getKind(), rewriter, isInt);
    rewriter.replaceOp(contractOp, {*result});
    return success();
  }

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
  FilterConstraintType filter;
};

/// Progressive lowering of ContractionOp.
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
/// to DOT or when other contraction patterns fail.
//
// TODO: break down into transpose/reshape/cast ops
//               when they become available to avoid code dup
// TODO: investigate lowering order impact on performance
LogicalResult
ContractionOpLowering::matchAndRewrite(vector::ContractionOp op,
                                       PatternRewriter &rewriter) const {
  if (failed(filter(op)))
    return failure();

  // TODO: support mixed mode contract lowering.
  if (op.getLhsType().getElementType() !=
          getElementTypeOrSelf(op.getAccType()) ||
      op.getRhsType().getElementType() != getElementTypeOrSelf(op.getAccType()))
    return failure();

  // TODO: the code below assumes the default contraction, make sure it supports
  // other kinds before enabling this lowering.
  if (op.getKind() != vector::CombiningKind::ADD) {
    return rewriter.notifyMatchFailure(
        op, "contractions other than 'add' not supported");
  }

  // TODO: implement benefits, cost models.
  MLIRContext *ctx = op.getContext();
  ContractionOpToMatmulOpLowering pat1(vectorTransformOptions, ctx);
  if (succeeded(pat1.matchAndRewrite(op, rewriter)))
    return success();
  ContractionOpToOuterProductOpLowering pat2(vectorTransformOptions, ctx);
  if (succeeded(pat2.matchAndRewrite(op, rewriter)))
    return success();
  ContractionOpToDotLowering pat3(vectorTransformOptions, ctx);
  if (succeeded(pat3.matchAndRewrite(op, rewriter)))
    return success();
  ContractOpToElementwise pat4(vectorTransformOptions, ctx);
  if (succeeded(pat4.matchAndRewrite(op, rewriter)))
    return success();

  // Vector mask setup.
  OpBuilder::InsertionGuard guard(rewriter);
  Operation *rootOp = op;
  Value mask;
  if (op.isMasked()) {
    rewriter.setInsertionPoint(op.getMaskingOp());
    rootOp = op.getMaskingOp();
    mask = op.getMaskingOp().getMask();
  }

  // Find first batch dimension in LHS/RHS, and lower when found.
  std::vector<std::pair<int64_t, int64_t>> batchDimMap = op.getBatchDimMap();
  if (!batchDimMap.empty()) {
    int64_t lhsIndex = batchDimMap[0].first;
    int64_t rhsIndex = batchDimMap[0].second;
    auto newOp = lowerParallel(rewriter, op, lhsIndex, rhsIndex, mask);
    if (failed(newOp))
      return failure();
    rewriter.replaceOp(rootOp, *newOp);
    return success();
  }

  // Collect contracting dimensions.
  std::vector<std::pair<int64_t, int64_t>> contractingDimMap =
      op.getContractingDimMap();
  DenseSet<int64_t> lhsContractingDimSet;
  DenseSet<int64_t> rhsContractingDimSet;
  for (auto &dimPair : contractingDimMap) {
    lhsContractingDimSet.insert(dimPair.first);
    rhsContractingDimSet.insert(dimPair.second);
  }

  // Find first free dimension in LHS, and lower when found.
  VectorType lhsType = op.getLhsType();
  for (int64_t lhsIndex = 0, e = lhsType.getRank(); lhsIndex < e; ++lhsIndex) {
    if (lhsContractingDimSet.count(lhsIndex) == 0) {
      auto newOp = lowerParallel(rewriter, op, lhsIndex, /*rhsIndex=*/-1, mask);
      if (failed(newOp))
        return failure();
      rewriter.replaceOp(rootOp, *newOp);
      return success();
    }
  }

  // Find first free dimension in RHS, and lower when found.
  VectorType rhsType = op.getRhsType();
  for (int64_t rhsIndex = 0, e = rhsType.getRank(); rhsIndex < e; ++rhsIndex) {
    if (rhsContractingDimSet.count(rhsIndex) == 0) {
      auto newOp = lowerParallel(rewriter, op, /*lhsIndex=*/-1, rhsIndex, mask);
      if (failed(newOp))
        return failure();
      rewriter.replaceOp(rootOp, *newOp);
      return success();
    }
  }

  // Lower the first remaining reduction dimension.
  if (!contractingDimMap.empty()) {
    auto newOp = lowerReduction(rewriter, op, mask);
    if (failed(newOp))
      return failure();
    rewriter.replaceOp(rootOp, *newOp);
    return success();
  }

  return failure();
}

// Lower one parallel dimension.
// Incidentally also tolerates unit-size (hence trivial) reduction dimensions.
// TODO: consider reusing existing contract unrolling
FailureOr<Value> ContractionOpLowering::lowerParallel(PatternRewriter &rewriter,
                                                      vector::ContractionOp op,
                                                      int64_t lhsIndex,
                                                      int64_t rhsIndex,
                                                      Value mask) const {
  VectorType lhsType = op.getLhsType();
  VectorType rhsType = op.getRhsType();
  VectorType resType = cast<VectorType>(op.getResultType());
  // Find the iterator type index and result index.
  SmallVector<AffineMap> iMap = op.getIndexingMapsArray();
  int64_t iterIndex = -1;
  int64_t dimSize = -1;
  if (lhsIndex >= 0) {
    iterIndex = iMap[0].getDimPosition(lhsIndex);
    if (rhsIndex >= 0 && iterIndex != iMap[1].getDimPosition(rhsIndex))
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "expected lhsIndex=" << lhsIndex << " and rhsIndex=" << rhsIndex
             << " to map to the same dimension";
      });
    if (lhsType.getScalableDims()[lhsIndex])
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "Unrolling scalable dimension (lhsIndex=" << lhsIndex
             << ") is not supported yet";
      });
    dimSize = lhsType.getDimSize(lhsIndex);
  } else if (rhsIndex >= 0) {
    iterIndex = iMap[1].getDimPosition(rhsIndex);
    if (rhsType.getScalableDims()[rhsIndex])
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "Unrolling scalable dimension (rhsIndex=" << rhsIndex
             << ") is not supported yet";
      });
    dimSize = rhsType.getDimSize(rhsIndex);
  }
  if (iterIndex < 0)
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "expected either lhsIndex=" << lhsIndex
           << " or rhsIndex=" << rhsIndex << " to be nonnegative";
    });
  // value_or(-1) means that we tolerate a dimension not appearing
  // in the result map. That can't happen for actual parallel iterators, but
  // the caller ContractionOpLowering::matchAndRewrite is currently calling
  // lowerParallel also for the case of unit-size reduction dims appearing only
  // on one of LHS or RHS, not both. At the moment, such cases are created by
  // CastAwayContractionLeadingOneDim, so we need to either support that or
  // modify that pattern.
  int64_t resIndex = getResultIndex(iMap[2], iterIndex).value_or(-1);
  if (resIndex == -1 && dimSize != 1)
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "expected the dimension for iterIndex=" << iterIndex
           << " to either appear in the result map, or to be a unit dimension";
    });

  // Construct new iterator types and affine map array attribute.
  std::array<AffineMap, 3> lowIndexingMaps = {
      adjustMap(iMap[0], iterIndex, rewriter),
      adjustMap(iMap[1], iterIndex, rewriter),
      adjustMap(iMap[2], iterIndex, rewriter)};
  auto lowAffine = rewriter.getAffineMapArrayAttr(lowIndexingMaps);
  auto lowIter =
      rewriter.getArrayAttr(adjustIter(op.getIteratorTypes(), iterIndex));
  // Unroll into a series of lower dimensional vector.contract ops.
  Location loc = op.getLoc();
  Value result = rewriter.create<arith::ConstantOp>(
      loc, resType, rewriter.getZeroAttr(resType));

  for (int64_t d = 0; d < dimSize; ++d) {
    auto lhs = reshapeLoad(loc, op.getLhs(), lhsType, lhsIndex, d, rewriter);
    auto rhs = reshapeLoad(loc, op.getRhs(), rhsType, rhsIndex, d, rewriter);
    auto acc = reshapeLoad(loc, op.getAcc(), resType, resIndex, d, rewriter);

    Value lowMask;
    if (mask)
      lowMask = reshapeLoad(loc, mask, cast<VectorType>(mask.getType()),
                            iterIndex, d, rewriter);

    Operation *lowContract = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, acc, lowAffine, lowIter);
    lowContract = maskOperation(rewriter, lowContract, lowMask);
    result = reshapeStore(loc, lowContract->getResult(0), result, resType,
                          resIndex, d, rewriter);
  }
  return result;
}

// Lower one reduction dimension.
FailureOr<Value> ContractionOpLowering::lowerReduction(
    PatternRewriter &rewriter, vector::ContractionOp op, Value mask) const {
  auto loc = op.getLoc();
  VectorType lhsType = op.getLhsType();
  VectorType rhsType = op.getRhsType();
  Type resType = op.getResultType();
  if (isa<VectorType>(resType))
    return rewriter.notifyMatchFailure(op,
                                       "did not expect a VectorType result");
  bool isInt = isa<IntegerType>(resType);
  // Use iterator index 0.
  int64_t iterIndex = 0;
  SmallVector<AffineMap> iMap = op.getIndexingMapsArray();
  std::optional<int64_t> lookupLhs = getResultIndex(iMap[0], iterIndex);
  std::optional<int64_t> lookupRhs = getResultIndex(iMap[1], iterIndex);
  if (!lookupLhs.has_value())
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "expected iterIndex=" << iterIndex << "to map to a LHS dimension";
    });
  if (!lookupRhs.has_value())
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "expected iterIndex=" << iterIndex << "to map to a RHS dimension";
    });
  int64_t lhsIndex = *lookupLhs;
  int64_t rhsIndex = *lookupRhs;
  int64_t dimSize = lhsType.getDimSize(lhsIndex);
  if (dimSize != rhsType.getDimSize(rhsIndex))
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "expect LHS dimension " << lhsIndex
           << " to have the same size as RHS dimension " << rhsIndex;
    });
  // Base case.
  if (lhsType.getRank() == 1) {
    if (rhsType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "When LHS has rank 1, expected also RHS to have rank 1");
    Value m = createMul(loc, op.getLhs(), op.getRhs(), isInt, rewriter);
    auto kind = vector::CombiningKind::ADD;

    Value acc = op.getAcc();
    Operation *reductionOp =
        acc ? rewriter.create<vector::ReductionOp>(loc, kind, m, acc)
            : rewriter.create<vector::ReductionOp>(loc, kind, m);
    return maskOperation(rewriter, reductionOp, mask)->getResult(0);
  }
  // Construct new iterator types and affine map array attribute.
  std::array<AffineMap, 3> lowIndexingMaps = {
      adjustMap(iMap[0], iterIndex, rewriter),
      adjustMap(iMap[1], iterIndex, rewriter),
      adjustMap(iMap[2], iterIndex, rewriter)};
  auto lowAffine = rewriter.getAffineMapArrayAttr(lowIndexingMaps);
  auto lowIter =
      rewriter.getArrayAttr(adjustIter(op.getIteratorTypes(), iterIndex));
  // Unroll into a series of lower dimensional vector.contract ops.
  // By feeding the initial accumulator into the first contraction,
  // and the result of each contraction into the next, eventually
  // the sum of all reductions is computed.
  Value result = op.getAcc();
  for (int64_t d = 0; d < dimSize; ++d) {
    auto lhs = reshapeLoad(loc, op.getLhs(), lhsType, lhsIndex, d, rewriter);
    auto rhs = reshapeLoad(loc, op.getRhs(), rhsType, rhsIndex, d, rewriter);
    Value newMask;
    if (mask)
      newMask = reshapeLoad(loc, mask, cast<VectorType>(mask.getType()),
                            iterIndex, d, rewriter);

    Operation *newContract = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, result, lowAffine, lowIter);
    result = maskOperation(rewriter, newContract, newMask)->getResult(0);
  }
  return result;
}

/// Progressive lowering of OuterProductOp.
/// One:
///   %x = vector.outerproduct %lhs, %rhs, %acc
/// is replaced by:
///   %z = zero-result
///   %0 = vector.extract %lhs[0]
///   %1 = vector.broadcast %0
///   %2 = vector.extract %acc[0]
///   %3 = vector.fma %1, %rhs, %2
///   %4 = vector.insert %3, %z[0]
///   ..
///   %x = vector.insert %.., %..[N-1]
///
class OuterProductOpLowering : public OpRewritePattern<vector::OuterProductOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::OuterProductOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resType = op.getResultVectorType();
    if ((resType.getShape().size() >= 2) && resType.allDimsScalable())
      return failure();

    auto loc = op.getLoc();

    VectorType lhsType = op.getOperandVectorTypeLHS();
    VectorType rhsType = dyn_cast<VectorType>(op.getOperandTypeRHS());
    Type eltType = resType.getElementType();
    bool isInt = isa<IntegerType, IndexType>(eltType);
    Value acc = op.getAcc();
    vector::CombiningKind kind = op.getKind();

    // Vector mask setup.
    OpBuilder::InsertionGuard guard(rewriter);
    auto maskableOp = cast<vector::MaskableOpInterface>(op.getOperation());
    Operation *rootOp;
    Value mask;
    if (maskableOp.isMasked()) {
      rewriter.setInsertionPoint(maskableOp.getMaskingOp());
      rootOp = maskableOp.getMaskingOp();
      mask = maskableOp.getMaskingOp().getMask();
    } else {
      rootOp = op;
    }

    if (!rhsType) {
      // Special case: AXPY operation.
      Value b = rewriter.create<vector::BroadcastOp>(loc, lhsType, op.getRhs());
      std::optional<Value> mult = createContractArithOp(
          loc, op.getLhs(), b, acc, kind, rewriter, isInt, mask);
      if (!mult.has_value())
        return failure();
      rewriter.replaceOp(rootOp, *mult);
      return success();
    }

    Value result = rewriter.create<arith::ConstantOp>(
        loc, resType, rewriter.getZeroAttr(resType));
    for (int64_t d = 0, e = resType.getDimSize(0); d < e; ++d) {
      Value x = rewriter.create<vector::ExtractOp>(loc, op.getLhs(), d);
      Value a = rewriter.create<vector::BroadcastOp>(loc, rhsType, x);
      Value r = nullptr;
      if (acc)
        r = rewriter.create<vector::ExtractOp>(loc, acc, d);
      Value extrMask;
      if (mask)
        extrMask = rewriter.create<vector::ExtractOp>(loc, mask, d);

      std::optional<Value> m = createContractArithOp(
          loc, a, op.getRhs(), r, kind, rewriter, isInt, extrMask);
      if (!m.has_value())
        return failure();
      result = rewriter.create<vector::InsertOp>(loc, *m, result, d);
    }

    rewriter.replaceOp(rootOp, result);
    return success();
  }
};

/// Progressively lower a `vector.contract %a, %b, %c` with row-major matmul
/// semantics to:
/// ```
///    %mta = maybe_transpose
///    %mtb = maybe_transpose
///    %flattened_a = vector.shape_cast %mta
///    %flattened_b = vector.shape_cast %mtb
///    %flattened_d = vector.matmul %flattened_a, %flattened_b
///    %mtd = vector.shape_cast %flattened_d
///    %d = maybe_untranspose %mtd
///    %e = add %c, %d
/// ```
/// `vector.matmul` later lowers to `llvm.matrix.multiply`.
//
/// This only kicks in when VectorTransformsOptions is set to `Matmul`.
/// vector.transpose operations are inserted if the vector.contract op is not a
/// row-major matrix multiply.
LogicalResult
ContractionOpToMatmulOpLowering::matchAndRewrite(vector::ContractionOp op,
                                                 PatternRewriter &rew) const {
  // TODO: Support vector.mask.
  auto maskableOp = cast<MaskableOpInterface>(op.getOperation());
  if (maskableOp.isMasked())
    return failure();

  if (vectorTransformOptions.vectorContractLowering !=
      vector::VectorContractLowering::Matmul)
    return failure();
  if (failed(filter(op)))
    return failure();

  auto iteratorTypes = op.getIteratorTypes().getValue();
  if (!isParallelIterator(iteratorTypes[0]) ||
      !isParallelIterator(iteratorTypes[1]) ||
      !isReductionIterator(iteratorTypes[2]))
    return failure();

  Type elementType = op.getLhsType().getElementType();
  if (!elementType.isIntOrFloat())
    return failure();

  Type dstElementType = op.getType();
  if (auto vecType = dyn_cast<VectorType>(dstElementType))
    dstElementType = vecType.getElementType();
  if (elementType != dstElementType)
    return failure();

  // Perform lhs + rhs transpositions to conform to matmul row-major semantics.
  // Bail out if the contraction cannot be put in this form.
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  AffineExpr m, n, k;
  bindDims(rew.getContext(), m, n, k);
  // LHS must be A(m, k) or A(k, m).
  Value lhs = op.getLhs();
  auto lhsMap = op.getIndexingMapsArray()[0];
  if (lhsMap == AffineMap::get(3, 0, {k, m}, ctx))
    lhs = rew.create<vector::TransposeOp>(loc, lhs, ArrayRef<int64_t>{1, 0});
  else if (lhsMap != AffineMap::get(3, 0, {m, k}, ctx))
    return failure();

  // RHS must be B(k, n) or B(n, k).
  Value rhs = op.getRhs();
  auto rhsMap = op.getIndexingMapsArray()[1];
  if (rhsMap == AffineMap::get(3, 0, {n, k}, ctx))
    rhs = rew.create<vector::TransposeOp>(loc, rhs, ArrayRef<int64_t>{1, 0});
  else if (rhsMap != AffineMap::get(3, 0, {k, n}, ctx))
    return failure();

  // At this point lhs and rhs are in row-major.
  VectorType lhsType = cast<VectorType>(lhs.getType());
  VectorType rhsType = cast<VectorType>(rhs.getType());
  int64_t lhsRows = lhsType.getDimSize(0);
  int64_t lhsColumns = lhsType.getDimSize(1);
  int64_t rhsColumns = rhsType.getDimSize(1);

  Type flattenedLHSType =
      VectorType::get(lhsType.getNumElements(), lhsType.getElementType());
  lhs = rew.create<vector::ShapeCastOp>(loc, flattenedLHSType, lhs);

  Type flattenedRHSType =
      VectorType::get(rhsType.getNumElements(), rhsType.getElementType());
  rhs = rew.create<vector::ShapeCastOp>(loc, flattenedRHSType, rhs);

  Value mul = rew.create<vector::MatmulOp>(loc, lhs, rhs, lhsRows, lhsColumns,
                                           rhsColumns);
  mul = rew.create<vector::ShapeCastOp>(
      loc,
      VectorType::get({lhsRows, rhsColumns},
                      getElementTypeOrSelf(op.getAcc().getType())),
      mul);

  // ACC must be C(m, n) or C(n, m).
  auto accMap = op.getIndexingMapsArray()[2];
  if (accMap == AffineMap::get(3, 0, {n, m}, ctx))
    mul = rew.create<vector::TransposeOp>(loc, mul, ArrayRef<int64_t>{1, 0});
  else if (accMap != AffineMap::get(3, 0, {m, n}, ctx))
    llvm_unreachable("invalid contraction semantics");

  Value res =
      isa<IntegerType>(elementType)
          ? static_cast<Value>(rew.create<arith::AddIOp>(loc, op.getAcc(), mul))
          : static_cast<Value>(
                rew.create<arith::AddFOp>(loc, op.getAcc(), mul));

  rew.replaceOp(op, res);
  return success();
}
} // namespace

void mlir::vector::populateVectorContractLoweringPatterns(
    RewritePatternSet &patterns, VectorTransformsOptions options,
    PatternBenefit benefit, bool disableOuterProductLowering) {
  if (!disableOuterProductLowering)
    patterns.add<OuterProductOpLowering>(patterns.getContext(), benefit);
  patterns.add<ContractionOpLowering, ContractionOpToMatmulOpLowering,
               ContractionOpToOuterProductOpLowering>(
      options, patterns.getContext(), benefit);
}

void mlir::vector::populateVectorOuterProductLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<OuterProductOpLowering>(patterns.getContext(), benefit);
}
