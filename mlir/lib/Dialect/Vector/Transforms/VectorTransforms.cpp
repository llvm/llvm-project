//===- VectorTransforms.cpp - Conversion within the Vector dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites as 1->N patterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"

#include <optional>
#include <type_traits>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/VectorInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "vector-to-vector"

using namespace mlir;
using namespace mlir::vector;

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
  Type lowType = VectorType::Builder(type).dropDim(0);
  // At extraction dimension?
  if (index == 0) {
    auto posAttr = rewriter.getI64ArrayAttr(pos);
    return rewriter.create<vector::ExtractOp>(loc, lowType, val, posAttr);
  }
  // Unroll leading dimensions.
  VectorType vType = lowType.cast<VectorType>();
  Type resType = VectorType::Builder(type).dropDim(index);
  auto resVectorType = resType.cast<VectorType>();
  Value result = rewriter.create<arith::ConstantOp>(
      loc, resVectorType, rewriter.getZeroAttr(resVectorType));
  for (int64_t d = 0, e = resVectorType.getDimSize(0); d < e; d++) {
    auto posAttr = rewriter.getI64ArrayAttr(d);
    Value ext = rewriter.create<vector::ExtractOp>(loc, vType, val, posAttr);
    Value load = reshapeLoad(loc, ext, vType, index - 1, pos, rewriter);
    result = rewriter.create<vector::InsertOp>(loc, resVectorType, load, result,
                                               posAttr);
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
  if (index == 0) {
    auto posAttr = rewriter.getI64ArrayAttr(pos);
    return rewriter.create<vector::InsertOp>(loc, type, val, result, posAttr);
  }
  // Unroll leading dimensions.
  Type lowType = VectorType::Builder(type).dropDim(0);
  VectorType vType = lowType.cast<VectorType>();
  Type insType = VectorType::Builder(vType).dropDim(0);
  for (int64_t d = 0, e = type.getDimSize(0); d < e; d++) {
    auto posAttr = rewriter.getI64ArrayAttr(d);
    Value ext = rewriter.create<vector::ExtractOp>(loc, vType, result, posAttr);
    Value ins = rewriter.create<vector::ExtractOp>(loc, insType, val, posAttr);
    Value sto = reshapeStore(loc, ins, ext, vType, index - 1, pos, rewriter);
    result = rewriter.create<vector::InsertOp>(loc, type, sto, result, posAttr);
  }
  return result;
}

template <typename IntType>
static SmallVector<IntType> extractVector(ArrayAttr arrayAttr) {
  return llvm::to_vector<4>(llvm::map_range(
      arrayAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
}

/// Helper to create arithmetic operation associated with a kind of contraction.
static std::optional<Value>
createContractArithOp(Location loc, Value x, Value y, Value acc,
                      vector::CombiningKind kind, PatternRewriter &rewriter,
                      bool isInt, Value mask = Value()) {
  using vector::CombiningKind;
  Value mul;

  if (isInt) {
    if (kind == CombiningKind::MINF || kind == CombiningKind::MAXF)
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
    if (acc && acc.getType().isa<VectorType>() && kind == CombiningKind::ADD) {
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

  return makeArithReduction(rewriter, loc, kind, mul, acc, mask);
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

namespace {

/// ShapeCastOpFolder folds cancelling ShapeCastOps away.
//
// Example:
//
//  The following MLIR with cancelling ShapeCastOps:
//
//   %0 = source : vector<5x4x2xf32>
//   %1 = shape_cast %0 : vector<5x4x2xf32> to vector<20x2xf32>
//   %2 = shape_cast %1 : vector<20x2xf32> to vector<5x4x2xf32>
//   %3 = user %2 : vector<5x4x2xf32>
//
//  Should canonicalize to the following:
//
//   %0 = source : vector<5x4x2xf32>
//   %1 = user %0 : vector<5x4x2xf32>
//
struct ShapeCastOpFolder : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                PatternRewriter &rewriter) const override {
    // Check if 'shapeCastOp' has vector source/result type.
    auto sourceVectorType =
        shapeCastOp.getSource().getType().dyn_cast_or_null<VectorType>();
    auto resultVectorType =
        shapeCastOp.getResult().getType().dyn_cast_or_null<VectorType>();
    if (!sourceVectorType || !resultVectorType)
      return failure();

    // Check if shape cast op source operand is also a shape cast op.
    auto sourceShapeCastOp = dyn_cast_or_null<vector::ShapeCastOp>(
        shapeCastOp.getSource().getDefiningOp());
    if (!sourceShapeCastOp)
      return failure();
    auto operandSourceVectorType =
        sourceShapeCastOp.getSource().getType().cast<VectorType>();
    auto operandResultVectorType = sourceShapeCastOp.getType();

    // Check if shape cast operations invert each other.
    if (operandSourceVectorType != resultVectorType ||
        operandResultVectorType != sourceVectorType)
      return failure();

    rewriter.replaceOp(shapeCastOp, sourceShapeCastOp.getSource());
    return success();
  }
};

/// Progressive lowering of BroadcastOp.
class BroadcastOpLowering : public OpRewritePattern<vector::BroadcastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    VectorType dstType = op.getResultVectorType();
    VectorType srcType = op.getSourceType().dyn_cast<VectorType>();
    Type eltType = dstType.getElementType();

    // Scalar to any vector can use splat.
    if (!srcType) {
      rewriter.replaceOpWithNewOp<vector::SplatOp>(op, dstType, op.getSource());
      return success();
    }

    // Determine rank of source and destination.
    int64_t srcRank = srcType.getRank();
    int64_t dstRank = dstType.getRank();

    // Stretching scalar inside vector (e.g. vector<1xf32>) can use splat.
    if (srcRank <= 1 && dstRank == 1) {
      Value ext;
      if (srcRank == 0)
        ext = rewriter.create<vector::ExtractElementOp>(loc, op.getSource());
      else
        ext = rewriter.create<vector::ExtractOp>(loc, op.getSource(), 0);
      rewriter.replaceOpWithNewOp<vector::SplatOp>(op, dstType, ext);
      return success();
    }

    // Duplicate this rank.
    // For example:
    //   %x = broadcast %y  : k-D to n-D, k < n
    // becomes:
    //   %b = broadcast %y  : k-D to (n-1)-D
    //   %x = [%b,%b,%b,%b] : n-D
    // becomes:
    //   %b = [%y,%y]       : (n-1)-D
    //   %x = [%b,%b,%b,%b] : n-D
    if (srcRank < dstRank) {
      // Duplication.
      VectorType resType =
          VectorType::get(dstType.getShape().drop_front(), eltType);
      Value bcst =
          rewriter.create<vector::BroadcastOp>(loc, resType, op.getSource());
      Value result = rewriter.create<arith::ConstantOp>(
          loc, dstType, rewriter.getZeroAttr(dstType));
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d)
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Find non-matching dimension, if any.
    assert(srcRank == dstRank);
    int64_t m = -1;
    for (int64_t r = 0; r < dstRank; r++)
      if (srcType.getDimSize(r) != dstType.getDimSize(r)) {
        m = r;
        break;
      }

    // All trailing dimensions are the same. Simply pass through.
    if (m == -1) {
      rewriter.replaceOp(op, op.getSource());
      return success();
    }

    // Any non-matching dimension forces a stretch along this rank.
    // For example:
    //   %x = broadcast %y : vector<4x1x2xf32> to vector<4x2x2xf32>
    // becomes:
    //   %a = broadcast %y[0] : vector<1x2xf32> to vector<2x2xf32>
    //   %b = broadcast %y[1] : vector<1x2xf32> to vector<2x2xf32>
    //   %c = broadcast %y[2] : vector<1x2xf32> to vector<2x2xf32>
    //   %d = broadcast %y[3] : vector<1x2xf32> to vector<2x2xf32>
    //   %x = [%a,%b,%c,%d]
    // becomes:
    //   %u = broadcast %y[0][0] : vector<2xf32> to vector <2x2xf32>
    //   %v = broadcast %y[1][0] : vector<2xf32> to vector <2x2xf32>
    //   %a = [%u, %v]
    //   ..
    //   %x = [%a,%b,%c,%d]
    VectorType resType =
        VectorType::get(dstType.getShape().drop_front(), eltType);
    Value result = rewriter.create<arith::ConstantOp>(
        loc, dstType, rewriter.getZeroAttr(dstType));
    if (m == 0) {
      // Stetch at start.
      Value ext = rewriter.create<vector::ExtractOp>(loc, op.getSource(), 0);
      Value bcst = rewriter.create<vector::BroadcastOp>(loc, resType, ext);
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d)
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
    } else {
      // Stetch not at start.
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d) {
        Value ext = rewriter.create<vector::ExtractOp>(loc, op.getSource(), d);
        Value bcst = rewriter.create<vector::BroadcastOp>(loc, resType, ext);
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Given a 'transpose' pattern, prune the rightmost dimensions that are not
/// transposed.
void pruneNonTransposedDims(ArrayRef<int64_t> transpose,
                            SmallVectorImpl<int64_t> &result) {
  size_t numTransposedDims = transpose.size();
  for (size_t transpDim : llvm::reverse(transpose)) {
    if (transpDim != numTransposedDims - 1)
      break;
    numTransposedDims--;
  }

  result.append(transpose.begin(), transpose.begin() + numTransposedDims);
}

/// Progressive lowering of TransposeOp.
/// One:
///   %x = vector.transpose %y, [1, 0]
/// is replaced by:
///   %z = arith.constant dense<0.000000e+00>
///   %0 = vector.extract %y[0, 0]
///   %1 = vector.insert %0, %z [0, 0]
///   ..
///   %x = vector.insert .., .. [.., ..]
class TransposeOpLowering : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  TransposeOpLowering(vector::VectorTransformsOptions vectorTransformOptions,
                      MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransposeOp>(context, benefit),
        vectorTransformOptions(vectorTransformOptions) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value input = op.getVector();
    VectorType inputType = op.getSourceVectorType();
    VectorType resType = op.getResultVectorType();

    // Set up convenience transposition table.
    SmallVector<int64_t> transp;
    for (auto attr : op.getTransp())
      transp.push_back(attr.cast<IntegerAttr>().getInt());

    if (vectorTransformOptions.vectorTransposeLowering ==
            vector::VectorTransposeLowering::Shuffle &&
        resType.getRank() == 2 && transp[0] == 1 && transp[1] == 0)
      return rewriter.notifyMatchFailure(
          op, "Options specifies lowering to shuffle");

    // Handle a true 2-D matrix transpose differently when requested.
    if (vectorTransformOptions.vectorTransposeLowering ==
            vector::VectorTransposeLowering::Flat &&
        resType.getRank() == 2 && transp[0] == 1 && transp[1] == 0) {
      Type flattenedType =
          VectorType::get(resType.getNumElements(), resType.getElementType());
      auto matrix =
          rewriter.create<vector::ShapeCastOp>(loc, flattenedType, input);
      auto rows = rewriter.getI32IntegerAttr(resType.getShape()[0]);
      auto columns = rewriter.getI32IntegerAttr(resType.getShape()[1]);
      Value trans = rewriter.create<vector::FlatTransposeOp>(
          loc, flattenedType, matrix, rows, columns);
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resType, trans);
      return success();
    }

    // Generate unrolled extract/insert ops. We do not unroll the rightmost
    // (i.e., highest-order) dimensions that are not transposed and leave them
    // in vector form to improve performance. Therefore, we prune those
    // dimensions from the shape/transpose data structures used to generate the
    // extract/insert ops.
    SmallVector<int64_t> prunedTransp;
    pruneNonTransposedDims(transp, prunedTransp);
    size_t numPrunedDims = transp.size() - prunedTransp.size();
    auto prunedInShape = inputType.getShape().drop_back(numPrunedDims);
    auto prunedInStrides = computeStrides(prunedInShape);

    // Generates the extract/insert operations for every scalar/vector element
    // of the leftmost transposed dimensions. We traverse every transpose
    // element using a linearized index that we delinearize to generate the
    // appropriate indices for the extract/insert operations.
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resType, rewriter.getZeroAttr(resType));
    int64_t numTransposedElements = ShapedType::getNumElements(prunedInShape);

    for (int64_t linearIdx = 0; linearIdx < numTransposedElements;
         ++linearIdx) {
      auto extractIdxs = delinearize(prunedInStrides, linearIdx);
      SmallVector<int64_t> insertIdxs(extractIdxs);
      applyPermutationToVector(insertIdxs, prunedTransp);
      Value extractOp =
          rewriter.create<vector::ExtractOp>(loc, input, extractIdxs);
      result =
          rewriter.create<vector::InsertOp>(loc, extractOp, result, insertIdxs);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
};

/// Rewrite a 2-D vector.transpose as a sequence of:
///   vector.shape_cast 2D -> 1D
///   vector.shuffle
///   vector.shape_cast 1D -> 2D
class TransposeOp2DToShuffleLowering
    : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  TransposeOp2DToShuffleLowering(
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransposeOp>(context, benefit),
        vectorTransformOptions(vectorTransformOptions) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType srcType = op.getSourceVectorType();
    if (srcType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "Not a 2D transpose");

    SmallVector<int64_t> transp;
    for (auto attr : op.getTransp())
      transp.push_back(attr.cast<IntegerAttr>().getInt());
    if (transp[0] != 1 && transp[1] != 0)
      return rewriter.notifyMatchFailure(op, "Not a 2D transpose permutation");

    if (vectorTransformOptions.vectorTransposeLowering !=
        VectorTransposeLowering::Shuffle)
      return rewriter.notifyMatchFailure(op, "Options do not ask for Shuffle");

    int64_t m = srcType.getShape().front(), n = srcType.getShape().back();
    Value casted = rewriter.create<vector::ShapeCastOp>(
        loc, VectorType::get({m * n}, srcType.getElementType()),
        op.getVector());
    SmallVector<int64_t> mask;
    mask.reserve(m * n);
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < m; ++i)
        mask.push_back(i * n + j);

    Value shuffled =
        rewriter.create<vector::ShuffleOp>(loc, casted, casted, mask);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        op, op.getResultVectorType(), shuffled);

    return success();
  }

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
};

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
    auto loc = op.getLoc();

    VectorType lhsType = op.getOperandVectorTypeLHS();
    VectorType rhsType = op.getOperandTypeRHS().dyn_cast<VectorType>();
    VectorType resType = op.getResultVectorType();
    Type eltType = resType.getElementType();
    bool isInt = eltType.isa<IntegerType, IndexType>();
    Value acc = (op.getAcc().empty()) ? nullptr : op.getAcc()[0];
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
      auto pos = rewriter.getI64ArrayAttr(d);
      Value x =
          rewriter.create<vector::ExtractOp>(loc, op.getLhs(), pos);
      Value a = rewriter.create<vector::BroadcastOp>(loc, rhsType, x);
      Value r = nullptr;
      if (acc)
        r = rewriter.create<vector::ExtractOp>(loc, acc, pos);
      Value extrMask;
      if (mask)
        extrMask = rewriter.create<vector::ExtractOp>(loc, mask, pos);

      std::optional<Value> m = createContractArithOp(
          loc, a, op.getRhs(), r, kind, rewriter, isInt, extrMask);
      if (!m.has_value())
        return failure();
      result = rewriter.create<vector::InsertOp>(loc, resType, *m, result, pos);
    }

    rewriter.replaceOp(rootOp, result);
    return success();
  }
};

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

    // TODO: Remove native masks from contraction op?
    if (!contractOp.getMasks().empty())
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
            contractOp.getResultType().cast<VectorType>().getDimSize(i));
        lhsTranspose.push_back(lhsDims.size() - 1);
      }
      std::optional<unsigned> rhsDim =
          getDimPosition(rhsMap, accMap.getDimPosition(i));
      if (rhsDim) {
        rhsTranspose.push_back(numRhsDimToBroadcast + *rhsDim);
      } else {
        // If the parallel dimension doesn't exist we will have to broadcast it.
        rhsDims.push_back(
            contractOp.getResultType().cast<VectorType>().getDimSize(i));
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
    newLhs = rewriter.create<vector::ExtractOp>(
        loc, newLhs, rewriter.getI64ArrayAttr(lhsOffsets));
    newRhs = rewriter.create<vector::ExtractOp>(
        loc, newRhs, rewriter.getI64ArrayAttr(rhsOffsets));
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

/// Progressive lowering of ConstantMaskOp.
/// One:
///   %x = vector.constant_mask [a,b]
/// is replaced by:
///   %z = zero-result
///   %l = vector.constant_mask [b]
///   %4 = vector.insert %l, %z[0]
///   ..
///   %x = vector.insert %l, %..[a-1]
/// until a one-dimensional vector is reached. All these operations
/// will be folded at LLVM IR level.
class ConstantMaskOpLowering : public OpRewritePattern<vector::ConstantMaskOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ConstantMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstType = op.getType();
    auto eltType = dstType.getElementType();
    auto dimSizes = op.getMaskDimSizes();
    int64_t rank = dstType.getRank();

    if (rank == 0) {
      assert(dimSizes.size() == 1 &&
             "Expected exactly one dim size for a 0-D vector");
      bool value = dimSizes[0].cast<IntegerAttr>().getInt() == 1;
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, dstType,
          DenseIntElementsAttr::get(
              VectorType::get(ArrayRef<int64_t>{}, rewriter.getI1Type()),
              ArrayRef<bool>{value}));
      return success();
    }

    // Scalable constant masks can only be lowered for the "none set" case.
    if (dstType.cast<VectorType>().isScalable()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, DenseElementsAttr::get(dstType, false));
      return success();
    }

    int64_t trueDim = std::min(dstType.getDimSize(0),
                               dimSizes[0].cast<IntegerAttr>().getInt());

    if (rank == 1) {
      // Express constant 1-D case in explicit vector form:
      //   [T,..,T,F,..,F].
      SmallVector<bool> values(dstType.getDimSize(0));
      for (int64_t d = 0; d < trueDim; d++)
        values[d] = true;
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, dstType, rewriter.getBoolVectorAttr(values));
      return success();
    }

    VectorType lowType =
        VectorType::get(dstType.getShape().drop_front(), eltType);
    SmallVector<int64_t> newDimSizes;
    for (int64_t r = 1; r < rank; r++)
      newDimSizes.push_back(dimSizes[r].cast<IntegerAttr>().getInt());
    Value trueVal = rewriter.create<vector::ConstantMaskOp>(
        loc, lowType, rewriter.getI64ArrayAttr(newDimSizes));
    Value result = rewriter.create<arith::ConstantOp>(
        loc, dstType, rewriter.getZeroAttr(dstType));
    for (int64_t d = 0; d < trueDim; d++) {
      auto pos = rewriter.getI64ArrayAttr(d);
      result =
          rewriter.create<vector::InsertOp>(loc, dstType, trueVal, result, pos);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Progressive lowering of CreateMaskOp.
/// One:
///   %x = vector.create_mask %a, ... : vector<dx...>
/// is replaced by:
///   %l = vector.create_mask ... : vector<...>  ; one lower rank
///   %0 = arith.cmpi "slt", %ci, %a       |
///   %1 = select %0, %l, %zeroes    |
///   %r = vector.insert %1, %pr [i] | d-times
///   %x = ....
/// until a one-dimensional vector is reached.
class CreateMaskOpLowering : public OpRewritePattern<vector::CreateMaskOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::CreateMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.getResult().getType().cast<VectorType>();
    int64_t rank = dstType.getRank();
    if (rank <= 1)
      return rewriter.notifyMatchFailure(
          op, "0-D and 1-D vectors are handled separately");

    auto loc = op.getLoc();
    auto eltType = dstType.getElementType();
    int64_t dim = dstType.getDimSize(0);
    Value idx = op.getOperand(0);

    VectorType lowType =
        VectorType::get(dstType.getShape().drop_front(), eltType);
    Value trueVal = rewriter.create<vector::CreateMaskOp>(
        loc, lowType, op.getOperands().drop_front());
    Value falseVal = rewriter.create<arith::ConstantOp>(
        loc, lowType, rewriter.getZeroAttr(lowType));
    Value result = rewriter.create<arith::ConstantOp>(
        loc, dstType, rewriter.getZeroAttr(dstType));
    for (int64_t d = 0; d < dim; d++) {
      Value bnd =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(d));
      Value val = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 bnd, idx);
      Value sel = rewriter.create<arith::SelectOp>(loc, val, trueVal, falseVal);
      auto pos = rewriter.getI64ArrayAttr(d);
      result =
          rewriter.create<vector::InsertOp>(loc, dstType, sel, result, pos);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// ShapeOp 2D -> 1D downcast serves the purpose of flattening 2-D to 1-D
/// vectors progressively on the way to target llvm.matrix intrinsics.
/// This iterates over the most major dimension of the 2-D vector and performs
/// rewrites into:
///   vector.extract from 2-D + vector.insert_strided_slice offset into 1-D
class ShapeCastOp2DDownCastRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    if (sourceVectorType.getRank() != 2 || resultVectorType.getRank() != 1)
      return failure();

    auto loc = op.getLoc();
    Value desc = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    unsigned mostMinorVectorSize = sourceVectorType.getShape()[1];
    for (int64_t i = 0, e = sourceVectorType.getShape().front(); i != e; ++i) {
      Value vec = rewriter.create<vector::ExtractOp>(loc, op.getSource(), i);
      desc = rewriter.create<vector::InsertStridedSliceOp>(
          loc, vec, desc,
          /*offsets=*/i * mostMinorVectorSize, /*strides=*/1);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

/// ShapeOp 1D -> 2D upcast serves the purpose of unflattening 2-D from 1-D
/// vectors progressively.
/// This iterates over the most major dimension of the 2-D vector and performs
/// rewrites into:
///   vector.extract_strided_slice from 1-D + vector.insert into 2-D
/// Note that 1-D extract_strided_slice are lowered to efficient vector.shuffle.
class ShapeCastOp2DUpCastRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    if (sourceVectorType.getRank() != 1 || resultVectorType.getRank() != 2)
      return failure();

    auto loc = op.getLoc();
    Value desc = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    unsigned mostMinorVectorSize = resultVectorType.getShape()[1];
    for (int64_t i = 0, e = resultVectorType.getShape().front(); i != e; ++i) {
      Value vec = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, op.getSource(), /*offsets=*/i * mostMinorVectorSize,
          /*sizes=*/mostMinorVectorSize,
          /*strides=*/1);
      desc = rewriter.create<vector::InsertOp>(loc, vec, desc, i);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

// We typically should not lower general shape cast operations into data
// movement instructions, since the assumption is that these casts are
// optimized away during progressive lowering. For completeness, however,
// we fall back to a reference implementation that moves all elements
// into the right place if we get here.
class ShapeCastOpRewritePattern : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();

    // Special case 2D/1D lowerings with better implementations.
    // TODO: make is ND/1D to allow generic ND->1D->MD.
    int64_t srcRank = sourceVectorType.getRank();
    int64_t resRank = resultVectorType.getRank();
    if ((srcRank == 2 && resRank == 1) || (srcRank == 1 && resRank == 2))
      return failure();

    // Generic ShapeCast lowering path goes all the way down to unrolled scalar
    // extract/insert chains.
    // TODO: consider evolving the semantics to only allow 1D source or dest and
    // drop this potentially very expensive lowering.
    // Compute number of elements involved in the reshape.
    int64_t numElts = 1;
    for (int64_t r = 0; r < srcRank; r++)
      numElts *= sourceVectorType.getDimSize(r);
    // Replace with data movement operations:
    //    x[0,0,0] = y[0,0]
    //    x[0,0,1] = y[0,1]
    //    x[0,1,0] = y[0,2]
    // etc., incrementing the two index vectors "row-major"
    // within the source and result shape.
    SmallVector<int64_t> srcIdx(srcRank);
    SmallVector<int64_t> resIdx(resRank);
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    for (int64_t i = 0; i < numElts; i++) {
      if (i != 0) {
        incIdx(srcIdx, sourceVectorType, srcRank - 1);
        incIdx(resIdx, resultVectorType, resRank - 1);
      }
      Value e = rewriter.create<vector::ExtractOp>(loc, op.getSource(), srcIdx);
      result = rewriter.create<vector::InsertOp>(loc, e, result, resIdx);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static void incIdx(SmallVector<int64_t> &idx, VectorType tp, int64_t r) {
    assert(0 <= r && r < tp.getRank());
    if (++idx[r] == tp.getDimSize(r)) {
      idx[r] = 0;
      incIdx(idx, tp, r - 1);
    }
  }
};

/// Convert MulIOp/MulFOp + MultiDimReductionOp<add> into ContractionOp.
/// Ex:
/// ```
///   %0 = arith.mulf %arg0, %arg1 : vector<8x32x16xf32>
///   %1 = vector.multi_reduction add, %0 [1]
///     : vector<8x32x16xf32> to vector<8x16xf32>
/// ```
/// Gets converted to:
/// ```
///   %1 = vector.contract {indexing_maps = [
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel", "reduction"],
///    kind = add} %0, %arg1, %cst_f0
///    : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
///  ```
struct MultiReduceToContract
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.getKind() != vector::CombiningKind::ADD)
      return failure();
    Operation *mulOp = reduceOp.getSource().getDefiningOp();
    if (!mulOp || !isa<arith::MulIOp, arith::MulFOp>(mulOp))
      return failure();
    SmallVector<bool> reductionMask = reduceOp.getReductionMask();
    auto srcMap = rewriter.getMultiDimIdentityMap(reductionMask.size());
    SmallVector<AffineExpr> exprs;
    SmallVector<vector::IteratorType> iteratorTypes;
    for (const auto &isReduceDim : llvm::enumerate(reductionMask)) {
      if (!isReduceDim.value()) {
        iteratorTypes.push_back(vector::IteratorType::parallel);
        exprs.push_back(rewriter.getAffineDimExpr(isReduceDim.index()));
      } else {
        iteratorTypes.push_back(vector::IteratorType::reduction);
      }
    }
    auto dstMap = AffineMap::get(/*dimCount=*/reductionMask.size(),
                                 /*symCount=*/0, exprs, reduceOp.getContext());
    rewriter.replaceOpWithNewOp<mlir::vector::ContractionOp>(
        reduceOp, mulOp->getOperand(0), mulOp->getOperand(1), reduceOp.getAcc(),
        rewriter.getAffineMapArrayAttr({srcMap, srcMap, dstMap}),
        rewriter.getArrayAttr(llvm::to_vector(llvm::map_range(
            iteratorTypes, [&](IteratorType t) -> mlir::Attribute {
              return IteratorTypeAttr::get(rewriter.getContext(), t);
            }))));
    return success();
  }
};

/// Merge LHS/RHS (A/B) TransposeOp into ContractionOp user.
/// Ex:
/// ```
///   %0 = vector.transpose %arg0, [2, 0, 1]
///     : vector<32x16x8xf32> to vector<8x32x16xf32>
///   %1 = vector.contract {indexing_maps = [
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel", "reduction"],
///    kind = add} %0, %arg1, %cst_f0
///    : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
/// ```
/// Gets converted to:
/// ```
///   %1 = vector.contract {indexing_maps = [
///         affine_map<(d0, d1, d2) -> (d1, d2, d0)>,
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel", "reduction"],
///    kind = add} %arg0, %arg1, %cst_f0
///    : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
///  ```
struct CombineContractABTranspose final
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap> maps =
        llvm::to_vector<4>(contractOp.getIndexingMapsArray());
    Value lhs = contractOp.getLhs();
    Value rhs = contractOp.getRhs();
    size_t index = 0;
    bool changed = false;
    for (Value *operand : {&lhs, &rhs}) {
      AffineMap &map = maps[index++];
      auto transposeOp = operand->getDefiningOp<vector::TransposeOp>();
      if (!transposeOp)
        continue;
      AffineMap permutationMap = AffineMap::getPermutationMap(
          extractVector<unsigned>(transposeOp.getTransp()),
          contractOp.getContext());
      map = inversePermutation(permutationMap).compose(map);
      *operand = transposeOp.getVector();
      changed = true;
    }
    if (!changed)
      return failure();
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, lhs, rhs, contractOp.getAcc(),
        rewriter.getAffineMapArrayAttr(maps), contractOp.getIteratorTypes());
    return success();
  }
};

/// Merges accumulator and result transposes into contract.
///
/// For example:
/// ```mlir
/// %accT = vector.transpose %acc, [0, 2, 1]
///   : vector<2x8x4xf32> to vector<2x4x8xf32>
/// %contract = vector.contract {
///   indexing_maps = [
///     affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
///     affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
///     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
///   ],
///   iterator_types = ["parallel", "parallel", "parallel", "reduction"],
///   kind = #vector.kind<add>
/// } %lhs, %rhs, %accT
///   : vector<2x4x4xf32>, vector<4x8xf32> into vector<2x4x8xf32>
/// %0 = vector.transpose %contract, [0, 2, 1]
///   : vector<2x4x8xf32> to vector<2x8x4>
/// ```
/// Becomes:
/// ```mlir
/// %0 = vector.contract {
///   indexing_maps = [
///     affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
///     affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
///     affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
///   ],
///   iterator_types = ["parallel", "parallel", "parallel", "reduction"],
///   kind = #vector.kind<add>
/// } %lhs, %rhs, %acc
///   : vector<2x4x4xf32>, vector<4x8xf32> into vector<2x8x4xf32>
/// ```
struct CombineContractResultTranspose final
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp resTOp,
                                PatternRewriter &rewriter) const override {
    auto contractOp = resTOp.getVector().getDefiningOp<vector::ContractionOp>();
    if (!contractOp || !contractOp->hasOneUse())
      return failure();

    auto accTOp = contractOp.getAcc().getDefiningOp<vector::TransposeOp>();
    if (!accTOp)
      return failure();

    MLIRContext *context = contractOp.getContext();
    auto maps = llvm::to_vector<3>(contractOp.getIndexingMapsArray());
    AffineMap contractMap = maps.back();

    // Accumulator transpose performs f(A) -> B. Contract performs g(C) -> B.
    // To index into A in contract, we need revert(f)(g(C)) -> A.
    auto accTMap = AffineMap::getPermutationMap(
        extractVector<unsigned>(accTOp.getTransp()), context);

    // Contract performs g(C) -> D. Result transpose performs h(D) -> E.
    // To index into E in contract, we need h(g(C)) -> E.
    auto resTMap = AffineMap::getPermutationMap(
        extractVector<unsigned>(resTOp.getTransp()), context);
    auto combinedResMap = resTMap.compose(contractMap);

    // The accumulator and result share the same indexing map. So they should be
    // the same to be able to merge. This means combinedResMap is the same as
    // inversePermutation(accTMap).compose(contractMap), which means
    if (inversePermutation(accTMap) != resTMap)
      return failure();
    maps.back() = combinedResMap;

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        resTOp, contractOp.getLhs(), contractOp.getRhs(), accTOp.getVector(),
        rewriter.getAffineMapArrayAttr(maps), contractOp.getIteratorTypes());
    return success();
  }
};

/// Merge BroadcastOp into ContractionOp user.
/// Ex:
/// ```
///   %0 = vector.broadcast %arg0 : vector<32x16xf32> to vector<8x32x16xf32>
///   %1 = vector.contract {indexing_maps = [
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel", "reduction"],
///    kind = add} %0, %arg1, %cst_f0
///    : vector<8x32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
/// ```
/// Gets converted to:
/// ```
///   %1 = vector.contract {indexing_maps = [
///         affine_map<(d0, d1, d2) -> (d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///         affine_map<(d0, d1, d2) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel", "reduction"],
///    kind = add} %arg0, %arg1, %cst_f0
///    : vector<32x16xf32>, vector<8x32x16xf32> into vector<8x32xf32>
///  ```
struct CombineContractBroadcast
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap> maps =
        llvm::to_vector<4>(contractOp.getIndexingMapsArray());
    Value lhs = contractOp.getLhs();
    Value rhs = contractOp.getRhs();
    size_t index = 0;
    bool changed = false;
    for (Value *operand : {&lhs, &rhs}) {
      AffineMap &map = maps[index++];
      auto broadcast = operand->getDefiningOp<vector::BroadcastOp>();
      if (!broadcast)
        continue;
      // contractionOp can only take vector as operands.
      auto srcType = broadcast.getSourceType().dyn_cast<VectorType>();
      if (!srcType ||
          srcType.getRank() == broadcast.getResultVectorType().getRank())
        continue;
      int64_t rankDiff =
          broadcast.getResultVectorType().getRank() - srcType.getRank();
      bool innerDimBroadcast = false;
      SmallVector<AffineExpr> originalDims;
      for (const auto &dim : llvm::enumerate(srcType.getShape())) {
        if (dim.value() != broadcast.getResultVectorType().getDimSize(
                               rankDiff + dim.index())) {
          innerDimBroadcast = true;
          break;
        }
        originalDims.push_back(
            rewriter.getAffineDimExpr(dim.index() + rankDiff));
      }
      // Contract doesn't support inner dimension broadcast. Once this is
      // relaxed we can remove this case.
      if (innerDimBroadcast)
        continue;

      // It would be incorrect to fold a broadcast onto a reduction dimension
      // of non-unit size.
      bool nonUnitDimReductionBroadcast = false;
      for (int64_t i = 0; i < rankDiff; ++i) {
        if (broadcast.getResultVectorType().getDimSize(i) != 1 &&
            isReductionIterator(contractOp.getIteratorTypes()
                                    .getValue()[map.getDimPosition(i)])) {
          nonUnitDimReductionBroadcast = true;
          break;
        }
      }
      if (nonUnitDimReductionBroadcast)
        continue;

      AffineMap broadcastMap =
          AffineMap::get(broadcast.getResultVectorType().getRank(), 0,
                         originalDims, contractOp.getContext());
      map = broadcastMap.compose(map);
      *operand = broadcast.getSource();
      changed = true;
    }

    if (!changed)
      return failure();

    // Determine which dims are usused, now that the maps have been composed
    // with the broadcast maps.
    llvm::SmallBitVector unusedDimsBitVector = getUnusedDimsBitVector(maps);
    // Compress unused dims.
    for (auto &m : maps)
      m = compressDims(m, unusedDimsBitVector);
    // Compute the combined iterators.
    SmallVector<Attribute> iterators;
    for (unsigned i = 0; i < unusedDimsBitVector.size(); ++i) {
      if (!unusedDimsBitVector.test(i))
        iterators.push_back(contractOp.getIteratorTypes().getValue()[i]);
    }
    // Check that compressing unused dims isn't removing all reduction dimension
    // pairs. For example, if the vector.contract had only one reduction
    // iterator and that was a unit-dimension created by a broadcast,
    // then we should bail here, otherwise we would create a contract without
    // a reduction dimension pair.
    bool hasReductionIteratorApplyingOnBothSides = false;
    for (unsigned i = 0; i < iterators.size(); ++i) {
      if (!isReductionIterator(iterators[i]))
        continue;
      if (getResultIndex(maps[0], i) && getResultIndex(maps[1], i)) {
        hasReductionIteratorApplyingOnBothSides = true;
        break;
      }
    }
    if (!hasReductionIteratorApplyingOnBothSides)
      return failure();

    // If the compressed maps have a dimension that is not used by either LHS or
    // RHS then the ContractionOp verifier would fail.
    if (getUnusedDimsBitVector({maps[0], maps[1]}).any())
      return failure();
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, lhs, rhs, contractOp.getAcc(),
        rewriter.getAffineMapArrayAttr(maps), rewriter.getArrayAttr(iterators));
    return success();
  }
};

/// Reorders cast(broadcast) to broadcast(cast). This makes broadcast ops and
/// contraction ops closer, which kicks in CombineContractBroadcast pattern when
/// casting ops are around these operations.
/// Ex:
/// ```
///   %0 = vector.broadcast %arg0 : vector<32x16xi8> to vector<8x32x16xi8>
///   %1 = arith.extsi %0 : vector<8x32x16xi8> to vector<8x32x16xi32>
/// ```
/// Gets converted to:
/// ```
///   %0 = arith.extsi %0 : vector<32x16xi8> to vector<32x16xi32>
///   %1 = vector.broadcast %arg0 : vector<32x16xi32> to vector<8x32x16xi32>
/// ```
struct ReorderCastOpsOnBroadcast
    : public OpInterfaceRewritePattern<CastOpInterface> {
  using OpInterfaceRewritePattern<CastOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CastOpInterface op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1)
      return failure();
    auto bcastOp = op->getOperand(0).getDefiningOp<vector::BroadcastOp>();
    if (!bcastOp)
      return failure();

    Type castResTy = getElementTypeOrSelf(op->getResult(0));
    if (auto vecTy = bcastOp.getSourceType().dyn_cast<VectorType>())
      castResTy = VectorType::get(vecTy.getShape(), castResTy);
    auto *castOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        bcastOp.getSource(), castResTy, op->getAttrs());
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        op, op->getResult(0).getType(), castOp->getResult(0));
    return success();
  }
};

/// Reorders elementwise(transpose) to transpose(elementwise). This makes
/// transpose ops and contraction ops closer, which kicks in
/// CombineContractABTranspose pattern when elementwise ops are between these
/// operations. Ex:
/// ```
/// %at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
/// %bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
/// %r = arith.addf %at, %bt : vector<2x4xf32>
/// ```
/// Gets converted to:
/// ```
/// %0 = arith.addf %a, %b : vector<4x2xf32>
/// %r = vector.transpose %0, [1, 0] : vector<2x4xf32>
/// ```
struct ReorderElementwiseOpsOnTranspose final
    : public OpTraitRewritePattern<OpTrait::Elementwise> {
  using OpTraitRewritePattern::OpTraitRewritePattern;
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumRegions() != 0)
      return failure();

    // Make sure all operands are transpose/constant ops and collect their
    // transposition maps.
    SmallVector<ArrayAttr> transposeMaps;
    transposeMaps.reserve(op->getNumOperands());
    // Record the initial type before transposition. We'll use its shape later.
    // Any type will do here as we will check all transpose maps are the same.
    VectorType srcType;
    for (Value operand : op->getOperands()) {
      auto transposeOp = operand.getDefiningOp<vector::TransposeOp>();
      if (transposeOp) {
        transposeMaps.push_back(transposeOp.getTransp());
        srcType = transposeOp.getSourceVectorType();
      } else if (!matchPattern(operand, m_Constant())) {
        return failure();
      }
    }
    if (transposeMaps.empty())
      return failure();
    // This is an elementwise op, so all transposed operands should have the
    // same type. We need to additionally check that all transposes uses the
    // same map.
    if (!llvm::all_equal(transposeMaps))
      return rewriter.notifyMatchFailure(op, "different transpose map");

    SmallVector<Value> srcValues;
    srcValues.reserve(op->getNumOperands());

    // If there are constant operands, we need to insert inverse transposes for
    // them. Calculate the inverse order first.
    auto order = extractVector<unsigned>(transposeMaps.front());
    SmallVector<int64_t> invOrder(order.size());
    for (int i = 0, e = order.size(); i < e; ++i)
      invOrder[order[i]] = i;

    for (Value operand : op->getOperands()) {
      auto transposeOp = operand.getDefiningOp<vector::TransposeOp>();
      if (transposeOp) {
        srcValues.push_back(transposeOp.getVector());
      } else {
        // This is a constant. Create a reverse transpose op for it.
        auto vectorType = VectorType::get(
            srcType.getShape(),
            operand.getType().cast<VectorType>().getElementType());
        srcValues.push_back(rewriter.create<vector::TransposeOp>(
            operand.getLoc(), vectorType, operand,
            rewriter.getI64ArrayAttr(invOrder)));
      }
    }

    auto vectorType = VectorType::get(
        srcType.getShape(),
        op->getResultTypes()[0].cast<VectorType>().getElementType());
    Operation *elementwiseOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), srcValues,
                        vectorType, op->getAttrs());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(
        op, op->getResultTypes()[0], elementwiseOp->getResult(0),
        transposeMaps.front());
    return success();
  }
};

} // namespace

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

namespace mlir {

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

  // TODO: Remove native masks from contraction op?
  if (!op.getMasks().empty())
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
  if (auto vecType = dstElementType.dyn_cast<VectorType>())
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
  VectorType lhsType = lhs.getType().cast<VectorType>();
  VectorType rhsType = rhs.getType().cast<VectorType>();
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
      elementType.isa<IntegerType>()
          ? static_cast<Value>(rew.create<arith::AddIOp>(loc, op.getAcc(), mul))
          : static_cast<Value>(
                rew.create<arith::AddFOp>(loc, op.getAcc(), mul));

  rew.replaceOp(op, res);
  return success();
}

namespace {

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
    auto vecType = elementType.dyn_cast<VectorType>();
    if (vecType)
      elementType = vecType.getElementType();
    if (elementType == dstElementType)
      return v;
    Type promotedType = dstElementType;
    if (vecType)
      promotedType = VectorType::get(vecType.getShape(), promotedType);
    if (dstElementType.isa<FloatType>())
      return rewriter.create<arith::ExtFOp>(loc, promotedType, v);
    return rewriter.create<arith::ExtSIOp>(loc, promotedType, v);
  }

  FailureOr<Value> outerProd(Value lhs, Value rhs, Value res, int reductionSize,
                             std::optional<Value> maybeMask = std::nullopt) {
    assert(reductionSize > 0);
    // Incremental support for masking.
    if (mask && !maybeMask.has_value())
      return failure();

    Type resElementType = res.getType().cast<VectorType>().getElementType();
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
    // Classical row-major matmul:  Just permute the lhs.
    if (layout({{m, k}, {k, n}, {m, n}}))
      return outerProd(t(lhs), rhs, res, lhsType.getDimSize(1),
                       t(mask, {2, 0, 1}));
    // TODO: may be better to fail and use some vector<k> -> scalar reduction.
    if (layout({{m, k}, {n, k}, {m, n}})) {
      Value tlhs = t(lhs);
      return outerProd(tlhs, t(rhs), res, lhsType.getDimSize(1));
    }
    // No need to permute anything.
    if (layout({{k, m}, {k, n}, {m, n}}))
      return outerProd(lhs, rhs, res, lhsType.getDimSize(0));
    // Just permute the rhs.
    if (layout({{k, m}, {n, k}, {m, n}}))
      return outerProd(lhs, t(rhs), res, lhsType.getDimSize(0));
    // Transposed output: swap RHS and LHS.
    // Classical row-major matmul: permute the lhs.
    if (layout({{m, k}, {k, n}, {n, m}}))
      return outerProd(rhs, t(lhs), res, lhsType.getDimSize(1));
    // TODO: may be better to fail and use some vector<k> -> scalar reduction.
    if (layout({{m, k}, {n, k}, {n, m}})) {
      Value trhs = t(rhs);
      return outerProd(trhs, t(lhs), res, lhsType.getDimSize(1));
    }
    if (layout({{k, m}, {k, n}, {n, m}}))
      return outerProd(rhs, lhs, res, lhsType.getDimSize(0));
    if (layout({{k, m}, {n, k}, {n, m}}))
      return outerProd(t(rhs), lhs, res, lhsType.getDimSize(0));
    return failure();
  }

  /// One outer parallel, one inner reduction (matvec flavor)
  FailureOr<Value> matvec() {
    if (!iters({Par(), Red()}))
      return failure();
    AffineExpr m, k;
    bindDims(rewriter.getContext(), m, k);

    // Case mat-vec: transpose.
    if (layout({{m, k}, {k}, {m}}))
      return outerProd(t(lhs), rhs, res, lhsType.getDimSize(1), t(mask));
    // Case mat-trans-vec: ready to go.
    if (layout({{k, m}, {k}, {m}}))
      return outerProd(lhs, rhs, res, lhsType.getDimSize(0));
    // Case vec-mat: swap and transpose.
    if (layout({{k}, {m, k}, {m}}))
      return outerProd(t(rhs), lhs, res, lhsType.getDimSize(0));
    // Case vec-mat-trans: swap and ready to go.
    if (layout({{k}, {k, m}, {m}}))
      return outerProd(rhs, lhs, res, lhsType.getDimSize(0));
    return failure();
  }

  //
  // One outer reduction, one inner parallel (tmatvec flavor)
  //
  FailureOr<Value> tmatvec() {
    if (!iters({Red(), Par()}))
      return failure();
    AffineExpr k, m;
    bindDims(rewriter.getContext(), k, m);

    // Case mat-vec: transpose.
    if (layout({{m, k}, {k}, {m}}))
      return outerProd(t(lhs), rhs, res, lhsType.getDimSize(1));
    // Case mat-trans-vec: ready to go.
    if (layout({{k, m}, {k}, {m}}))
      return outerProd(lhs, rhs, res, lhsType.getDimSize(0));
    // Case vec-mat: swap and transpose.
    if (layout({{k}, {m, k}, {m}}))
      return outerProd(t(rhs), lhs, res, lhsType.getDimSize(0));
    // Case vec-mat-trans: swap and ready to go.
    if (layout({{k}, {k, m}, {m}}))
      return outerProd(rhs, lhs, res, lhsType.getDimSize(0));
    return failure();
  }

private:
  vector::CombiningKind kind;
  Value lhs, rhs, res, mask;
  VectorType lhsType;
};
} // namespace

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
  // TODO: Remove native masks from contraction op?
  if (!op.getMasks().empty())
    return failure();

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

  // TODO: Remove native masks from contraction op?
  if (!op.getMasks().empty())
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

  VectorType dstType = op.getResultType().cast<VectorType>();
  assert(dstType.getRank() >= 1 && dstType.getRank() <= 2 &&
         "Expected dst type of rank 1 or 2");

  unsigned rank = dstType.getRank();
  unsigned dstRows = dstType.getShape()[0];
  unsigned dstColumns = rank == 1 ? 1 : dstType.getShape()[1];

  // ExtractOp does not allow dynamic indexing, we must unroll explicitly.
  Value res = rewriter.create<arith::ConstantOp>(loc, dstType,
                                                 rewriter.getZeroAttr(dstType));
  bool isInt = dstType.getElementType().isa<IntegerType>();
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
  // TODO: Support vector.mask.
  auto maskableOp = cast<MaskableOpInterface>(op.getOperation());
  if (maskableOp.isMasked())
    return failure();

  // TODO: Remove native masks from contraction op?
  if (!op.getMasks().empty())
    return failure();

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

  // Find first batch dimension in LHS/RHS, and lower when found.
  std::vector<std::pair<int64_t, int64_t>> batchDimMap = op.getBatchDimMap();
  if (!batchDimMap.empty()) {
    int64_t lhsIndex = batchDimMap[0].first;
    int64_t rhsIndex = batchDimMap[0].second;
    auto newOp = lowerParallel(op, lhsIndex, rhsIndex, rewriter);
    if (failed(newOp))
      return failure();
    rewriter.replaceOp(op, *newOp);
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
      auto newOp = lowerParallel(op, lhsIndex, /*rhsIndex=*/-1, rewriter);
      if (failed(newOp))
        return failure();
      rewriter.replaceOp(op, *newOp);
      return success();
    }
  }

  // Find first free dimension in RHS, and lower when found.
  VectorType rhsType = op.getRhsType();
  for (int64_t rhsIndex = 0, e = rhsType.getRank(); rhsIndex < e; ++rhsIndex) {
    if (rhsContractingDimSet.count(rhsIndex) == 0) {
      auto newOp = lowerParallel(op, /*lhsIndex=*/-1, rhsIndex, rewriter);
      if (failed(newOp))
        return failure();
      rewriter.replaceOp(op, *newOp);
      return success();
    }
  }

  // Lower the first remaining reduction dimension.
  if (!contractingDimMap.empty()) {
    auto newOp = lowerReduction(op, rewriter);
    if (failed(newOp))
      return failure();
    rewriter.replaceOp(op, *newOp);
    return success();
  }

  return failure();
}

// Lower one parallel dimension.
// Incidentally also tolerates unit-size (hence trivial) reduction dimensions.
// TODO: consider reusing existing contract unrolling
FailureOr<Value>
ContractionOpLowering::lowerParallel(vector::ContractionOp op, int64_t lhsIndex,
                                     int64_t rhsIndex,
                                     PatternRewriter &rewriter) const {
  VectorType lhsType = op.getLhsType();
  VectorType rhsType = op.getRhsType();
  VectorType resType = op.getResultType().cast<VectorType>();
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
    dimSize = lhsType.getDimSize(lhsIndex);
  } else if (rhsIndex >= 0) {
    iterIndex = iMap[1].getDimPosition(rhsIndex);
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
    Value lowContract = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, acc, lowAffine, lowIter);
    result =
        reshapeStore(loc, lowContract, result, resType, resIndex, d, rewriter);
  }
  return result;
}

// Lower one reduction dimension.
FailureOr<Value>
ContractionOpLowering::lowerReduction(vector::ContractionOp op,
                                      PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  VectorType lhsType = op.getLhsType();
  VectorType rhsType = op.getRhsType();
  Type resType = op.getResultType();
  if (resType.isa<VectorType>())
    return rewriter.notifyMatchFailure(op,
                                       "did not expect a VectorType result");
  bool isInt = resType.isa<IntegerType>();
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
    if (auto acc = op.getAcc())
      return rewriter.create<vector::ReductionOp>(loc, kind, m, acc)
          .getResult();
    return rewriter.create<vector::ReductionOp>(loc, kind, m).getResult();
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
    result = rewriter.create<vector::ContractionOp>(loc, lhs, rhs, result,
                                                    lowAffine, lowIter);
  }
  return result;
}

} // namespace mlir

/// Progressive lowering of transfer_read. This pattern supports lowering of
/// `vector.transfer_read` to a combination of `vector.load` and
/// `vector.broadcast` if all of the following hold:
/// - Stride of most minor memref dimension must be 1.
/// - Out-of-bounds masking is not required.
/// - If the memref's element type is a vector type then it coincides with the
///   result type.
/// - The permutation map doesn't perform permutation (broadcasting is allowed).
struct TransferReadToVectorLoadLowering
    : public OpRewritePattern<vector::TransferReadOp> {
  TransferReadToVectorLoadLowering(MLIRContext *context,
                                   std::optional<unsigned> maxRank,
                                   PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransferReadOp>(context, benefit),
        maxTransferRank(maxRank) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    if (maxTransferRank && read.getVectorType().getRank() > *maxTransferRank)
      return failure();

    SmallVector<unsigned> broadcastedDims;
    // Permutations are handled by VectorToSCF or
    // populateVectorTransferPermutationMapLoweringPatterns.
    // We let the 0-d corner case pass-through as it is supported.
    if (!read.getPermutationMap().isMinorIdentityWithBroadcasting(
            &broadcastedDims))
      return failure();

    auto memRefType = read.getShapedType().dyn_cast<MemRefType>();
    if (!memRefType)
      return failure();

    // Non-unit strides are handled by VectorToSCF.
    if (!vector::isLastMemrefDimUnitStride(memRefType))
      return failure();

    // If there is broadcasting involved then we first load the unbroadcasted
    // vector, and then broadcast it with `vector.broadcast`.
    ArrayRef<int64_t> vectorShape = read.getVectorType().getShape();
    SmallVector<int64_t> unbroadcastedVectorShape(vectorShape.begin(),
                                                  vectorShape.end());
    for (unsigned i : broadcastedDims)
      unbroadcastedVectorShape[i] = 1;
    VectorType unbroadcastedVectorType = VectorType::get(
        unbroadcastedVectorShape, read.getVectorType().getElementType());

    // `vector.load` supports vector types as memref's elements only when the
    // resulting vector type is the same as the element type.
    auto memrefElTy = memRefType.getElementType();
    if (memrefElTy.isa<VectorType>() && memrefElTy != unbroadcastedVectorType)
      return failure();

    // Otherwise, element types of the memref and the vector must match.
    if (!memrefElTy.isa<VectorType>() &&
        memrefElTy != read.getVectorType().getElementType())
      return failure();

    // Out-of-bounds dims are handled by MaterializeTransferMask.
    if (read.hasOutOfBoundsDim())
      return failure();

    // Create vector load op.
    Operation *loadOp;
    if (read.getMask()) {
      Value fill = rewriter.create<vector::SplatOp>(
          read.getLoc(), unbroadcastedVectorType, read.getPadding());
      loadOp = rewriter.create<vector::MaskedLoadOp>(
          read.getLoc(), unbroadcastedVectorType, read.getSource(),
          read.getIndices(), read.getMask(), fill);
    } else {
      loadOp = rewriter.create<vector::LoadOp>(
          read.getLoc(), unbroadcastedVectorType, read.getSource(),
          read.getIndices());
    }

    // Insert a broadcasting op if required.
    if (!broadcastedDims.empty()) {
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
          read, read.getVectorType(), loadOp->getResult(0));
    } else {
      rewriter.replaceOp(read, loadOp->getResult(0));
    }

    return success();
  }

  std::optional<unsigned> maxTransferRank;
};

/// Replace a 0-d vector.load with a memref.load + vector.broadcast.
// TODO: we shouldn't cross the vector/scalar domains just for this
// but atm we lack the infra to avoid it. Possible solutions include:
// - go directly to LLVM + bitcast
// - introduce a bitcast op and likely a new pointer dialect
// - let memref.load/store additionally support the 0-d vector case
// There are still deeper data layout issues lingering even in this
// trivial case (for architectures for which this matters).
struct VectorLoadToMemrefLoadLowering
    : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = loadOp.getVectorType();
    if (vecType.getNumElements() != 1)
      return failure();
    auto memrefLoad = rewriter.create<memref::LoadOp>(
        loadOp.getLoc(), loadOp.getBase(), loadOp.getIndices());
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(loadOp, vecType,
                                                     memrefLoad);
    return success();
  }
};

/// Replace a 0-d vector.store with a vector.extractelement + memref.store.
struct VectorStoreToMemrefStoreLowering
    : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = storeOp.getVectorType();
    if (vecType.getNumElements() != 1)
      return failure();
    Value extracted;
    if (vecType.getRank() == 0) {
      // TODO: Unifiy once ExtractOp supports 0-d vectors.
      extracted = rewriter.create<vector::ExtractElementOp>(
          storeOp.getLoc(), storeOp.getValueToStore());
    } else {
      SmallVector<int64_t> indices(vecType.getRank(), 0);
      extracted = rewriter.create<vector::ExtractOp>(
          storeOp.getLoc(), storeOp.getValueToStore(), indices);
    }

    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        storeOp, extracted, storeOp.getBase(), storeOp.getIndices());
    return success();
  }
};

/// Progressive lowering of transfer_write. This pattern supports lowering of
/// `vector.transfer_write` to `vector.store` if all of the following hold:
/// - Stride of most minor memref dimension must be 1.
/// - Out-of-bounds masking is not required.
/// - If the memref's element type is a vector type then it coincides with the
///   type of the written value.
/// - The permutation map is the minor identity map (neither permutation nor
///   broadcasting is allowed).
struct TransferWriteToVectorStoreLowering
    : public OpRewritePattern<vector::TransferWriteOp> {
  TransferWriteToVectorStoreLowering(MLIRContext *context,
                                     std::optional<unsigned> maxRank,
                                     PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransferWriteOp>(context, benefit),
        maxTransferRank(maxRank) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    if (maxTransferRank && write.getVectorType().getRank() > *maxTransferRank)
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "rank exceeds maxTransferRank: " << write;
      });

    // Permutations are handled by VectorToSCF or
    // populateVectorTransferPermutationMapLoweringPatterns.
    if ( // pass-through for the 0-d corner case.
        !write.getPermutationMap().isMinorIdentity())
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "permutation map is not minor identity: " << write;
      });

    auto memRefType = write.getShapedType().dyn_cast<MemRefType>();
    if (!memRefType)
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "not a memref type: " << write;
      });

    // Non-unit strides are handled by VectorToSCF.
    if (!vector::isLastMemrefDimUnitStride(memRefType))
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "most minor stride is not 1: " << write;
      });

    // `vector.store` supports vector types as memref's elements only when the
    // type of the vector value being written is the same as the element type.
    auto memrefElTy = memRefType.getElementType();
    if (memrefElTy.isa<VectorType>() && memrefElTy != write.getVectorType())
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "elemental type mismatch: " << write;
      });

    // Otherwise, element types of the memref and the vector must match.
    if (!memrefElTy.isa<VectorType>() &&
        memrefElTy != write.getVectorType().getElementType())
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "elemental type mismatch: " << write;
      });

    // Out-of-bounds dims are handled by MaterializeTransferMask.
    if (write.hasOutOfBoundsDim())
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "out of bounds dim: " << write;
      });
    if (write.getMask()) {
      rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
          write, write.getSource(), write.getIndices(), write.getMask(),
          write.getVector());
    } else {
      rewriter.replaceOpWithNewOp<vector::StoreOp>(
          write, write.getVector(), write.getSource(), write.getIndices());
    }
    return success();
  }

  std::optional<unsigned> maxTransferRank;
};

// Returns the values in `arrayAttr` as an integer vector.
static SmallVector<int64_t> getIntValueVector(ArrayAttr arrayAttr) {
  return llvm::to_vector<4>(
      llvm::map_range(arrayAttr.getAsRange<IntegerAttr>(),
                      [](IntegerAttr attr) { return attr.getInt(); }));
}

// Shuffles vector.bitcast op after vector.extract op.
//
// This transforms IR like:
//   %0 = vector.bitcast %src : vector<4xf32> to vector<8xf16>
//   %1 = vector.extract %0[3] : vector<8xf16>
// Into:
//   %0 = vector.extract %src[1] : vector<4xf32>
//   %1 = vector.bitcast %0: vector<1xf32> to vector<2xf16>
//   %2 = vector.extract %1[1] : vector<2xf16>
struct BubbleDownVectorBitCastForExtract
    : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Only support extracting scalars for now.
    if (extractOp.getSourceVectorType().getRank() != 1)
      return failure();

    auto castOp = extractOp.getVector().getDefiningOp<vector::BitCastOp>();
    if (!castOp)
      return failure();

    VectorType castSrcType = castOp.getSourceVectorType();
    VectorType castDstType = castOp.getResultVectorType();
    assert(castSrcType.getRank() == castDstType.getRank());

    // Fail to match if we only have one element in the cast op source.
    // This is to avoid infinite loop given that this pattern can generate
    // such cases.
    if (castSrcType.getNumElements() == 1)
      return failure();

    // Only support casting to a larger number of elements or now.
    // E.g., vector<4xf32> -> vector<8xf16>.
    if (castSrcType.getNumElements() > castDstType.getNumElements())
      return failure();

    unsigned expandRatio =
        castDstType.getNumElements() / castSrcType.getNumElements();

    auto getFirstIntValue = [](ArrayAttr attr) -> uint64_t {
      return (*attr.getAsValueRange<IntegerAttr>().begin()).getZExtValue();
    };

    uint64_t index = getFirstIntValue(extractOp.getPosition());

    // Get the single scalar (as a vector) in the source value that packs the
    // desired scalar. E.g. extract vector<1xf32> from vector<4xf32>
    VectorType oneScalarType =
        VectorType::get({1}, castSrcType.getElementType());
    Value packedValue = rewriter.create<vector::ExtractOp>(
        extractOp.getLoc(), oneScalarType, castOp.getSource(),
        rewriter.getI64ArrayAttr(index / expandRatio));

    // Cast it to a vector with the desired scalar's type.
    // E.g. f32 -> vector<2xf16>
    VectorType packedType =
        VectorType::get({expandRatio}, castDstType.getElementType());
    Value castedValue = rewriter.create<vector::BitCastOp>(
        extractOp.getLoc(), packedType, packedValue);

    // Finally extract the desired scalar.
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(
        extractOp, extractOp.getType(), castedValue,
        rewriter.getI64ArrayAttr(index % expandRatio));

    return success();
  }
};

// Shuffles vector.bitcast op after vector.extract_strided_slice op.
//
// This transforms IR like:
//    %cast = vector.bitcast %arg0: vector<4xf32> to vector<8xf16>
//     %0 = vector.extract_strided_slice %cast {
//            offsets = [4], sizes = [4], strides = [1]
//          } : vector<8xf16> to vector<4xf16>
// Into:
//   %0 = vector.extract_strided_slice %src {
//          offsets = [2], sizes = [2], strides = [1]
//        } : vector<4xf32> to vector<2xf32>
//   %1 = vector.bitcast %0 : vector<2xf32> to vector<4xf16>
struct BubbleDownBitCastForStridedSliceExtract
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto castOp = extractOp.getVector().getDefiningOp<vector::BitCastOp>();
    if (!castOp)
      return failure();

    VectorType castSrcType = castOp.getSourceVectorType();
    VectorType castDstType = castOp.getResultVectorType();
    assert(castSrcType.getRank() == castDstType.getRank());

    int64_t castSrcLastDim = castSrcType.getShape().back();
    int64_t castDstLastDim = castDstType.getShape().back();
    // Require casting to more elements for now; other cases to be implemented.
    if (castSrcLastDim > castDstLastDim)
      return failure();

    // Only accept all one strides for now.
    if (llvm::any_of(extractOp.getStrides().getAsValueRange<IntegerAttr>(),
                     [](const APInt &val) { return !val.isOne(); }))
      return failure();

    unsigned rank = extractOp.getSourceVectorType().getRank();
    assert(castDstLastDim % castSrcLastDim == 0);
    int64_t expandRatio = castDstLastDim / castSrcLastDim;

    // If we have a less number of offsets than the rank, then implicitly we
    // are selecting the full range for the last bitcasted dimension; other
    // dimensions aren't affected. Otherwise, we need to scale down the last
    // dimension's offset given we are extracting from less elements now.
    ArrayAttr newOffsets = extractOp.getOffsets();
    if (newOffsets.size() == rank) {
      SmallVector<int64_t> offsets = getIntValueVector(newOffsets);
      if (offsets.back() % expandRatio != 0)
        return failure();
      offsets.back() = offsets.back() / expandRatio;
      newOffsets = rewriter.getI64ArrayAttr(offsets);
    }

    // Similarly for sizes.
    ArrayAttr newSizes = extractOp.getSizes();
    if (newSizes.size() == rank) {
      SmallVector<int64_t> sizes = getIntValueVector(newSizes);
      if (sizes.back() % expandRatio != 0)
        return failure();
      sizes.back() = sizes.back() / expandRatio;
      newSizes = rewriter.getI64ArrayAttr(sizes);
    }

    SmallVector<int64_t> dims =
        llvm::to_vector<4>(extractOp.getType().cast<VectorType>().getShape());
    dims.back() = dims.back() / expandRatio;
    VectorType newExtractType =
        VectorType::get(dims, castSrcType.getElementType());

    auto newExtractOp = rewriter.create<vector::ExtractStridedSliceOp>(
        extractOp.getLoc(), newExtractType, castOp.getSource(), newOffsets,
        newSizes, extractOp.getStrides());

    rewriter.replaceOpWithNewOp<vector::BitCastOp>(
        extractOp, extractOp.getType(), newExtractOp);

    return success();
  }
};

// Shuffles vector.bitcast op before vector.insert_strided_slice op.
//
// This transforms IR like:
//   %0 = vector.insert_strided_slice %src, %dst {
//          offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
//   %1 = vector.bitcast %0: vector<8xf16> to vector<4xf32>
// Into:
//   %0 = vector.bitcast %src : vector<4xf16> to vector<2xf32>
//   %1 = vector.bitcast %dst : vector<8xf16> to vector<4xf32>
//   %2 = vector.insert_strided_slice %src, %dst {
//          offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>
struct BubbleUpBitCastForStridedSliceInsert
    : public OpRewritePattern<vector::BitCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    VectorType castSrcType = bitcastOp.getSourceVectorType();
    VectorType castDstType = bitcastOp.getResultVectorType();
    assert(castSrcType.getRank() == castDstType.getRank());
    // Skip 0-D vector which will not from InsertStridedSliceOp.
    if (castSrcType.getRank() == 0)
      return failure();

    int64_t castSrcLastDim = castSrcType.getShape().back();
    int64_t castDstLastDim = castDstType.getShape().back();
    // Require casting to less elements for now; other cases to be implemented.
    if (castSrcLastDim < castDstLastDim)
      return failure();

    assert(castSrcLastDim % castDstLastDim == 0);
    int64_t shrinkRatio = castSrcLastDim / castDstLastDim;

    auto insertOp =
        bitcastOp.getSource().getDefiningOp<vector::InsertStridedSliceOp>();
    if (!insertOp)
      return failure();

    // Only accept all one strides for now.
    if (llvm::any_of(insertOp.getStrides().getAsValueRange<IntegerAttr>(),
                     [](const APInt &val) { return !val.isOne(); }))
      return failure();

    unsigned rank = insertOp.getSourceVectorType().getRank();
    // Require insert op to have the same rank for the source and destination
    // vector; other cases to be implemented.
    if (rank != insertOp.getDestVectorType().getRank())
      return failure();

    // Requires that shape of insert op src is castable to dstType.
    unsigned sourceWidth = castSrcType.getElementType().getIntOrFloatBitWidth();
    unsigned destinationWidth =
        castDstType.getElementType().getIntOrFloatBitWidth();
    unsigned numElements = destinationWidth / sourceWidth;
    if (insertOp.getSourceVectorType().getNumElements() % numElements != 0)
      return failure();

    ArrayAttr newOffsets = insertOp.getOffsets();
    assert(newOffsets.size() == rank);
    SmallVector<int64_t> offsets = getIntValueVector(newOffsets);
    if (offsets.back() % shrinkRatio != 0)
      return failure();
    offsets.back() = offsets.back() / shrinkRatio;
    newOffsets = rewriter.getI64ArrayAttr(offsets);

    SmallVector<int64_t> srcDims =
        llvm::to_vector<4>(insertOp.getSourceVectorType().getShape());
    srcDims.back() = srcDims.back() / shrinkRatio;
    VectorType newCastSrcType =
        VectorType::get(srcDims, castDstType.getElementType());

    auto newCastSrcOp = rewriter.create<vector::BitCastOp>(
        bitcastOp.getLoc(), newCastSrcType, insertOp.getSource());

    SmallVector<int64_t> dstDims =
        llvm::to_vector<4>(insertOp.getDestVectorType().getShape());
    dstDims.back() = dstDims.back() / shrinkRatio;
    VectorType newCastDstType =
        VectorType::get(dstDims, castDstType.getElementType());

    auto newCastDstOp = rewriter.create<vector::BitCastOp>(
        bitcastOp.getLoc(), newCastDstType, insertOp.getDest());

    rewriter.replaceOpWithNewOp<vector::InsertStridedSliceOp>(
        bitcastOp, bitcastOp.getType(), newCastSrcOp, newCastDstOp, newOffsets,
        insertOp.getStrides());

    return success();
  }
};

// Helper that returns a vector comparison that constructs a mask:
//     mask = [0,1,..,n-1] + [o,o,..,o] < [b,b,..,b]
//
// If `dim == 0` then the result will be a 0-D vector.
//
// NOTE: The LLVM::GetActiveLaneMaskOp intrinsic would provide an alternative,
//       much more compact, IR for this operation, but LLVM eventually
//       generates more elaborate instructions for this intrinsic since it
//       is very conservative on the boundary conditions.
static Value buildVectorComparison(PatternRewriter &rewriter, Operation *op,
                                   bool force32BitVectorIndices, int64_t dim,
                                   Value b, Value *off = nullptr) {
  auto loc = op->getLoc();
  // If we can assume all indices fit in 32-bit, we perform the vector
  // comparison in 32-bit to get a higher degree of SIMD parallelism.
  // Otherwise we perform the vector comparison using 64-bit indices.
  Type idxType =
      force32BitVectorIndices ? rewriter.getI32Type() : rewriter.getI64Type();
  DenseIntElementsAttr indicesAttr;
  if (dim == 0 && force32BitVectorIndices) {
    indicesAttr = DenseIntElementsAttr::get(
        VectorType::get(ArrayRef<int64_t>{}, idxType), ArrayRef<int32_t>{0});
  } else if (dim == 0) {
    indicesAttr = DenseIntElementsAttr::get(
        VectorType::get(ArrayRef<int64_t>{}, idxType), ArrayRef<int64_t>{0});
  } else if (force32BitVectorIndices) {
    indicesAttr = rewriter.getI32VectorAttr(
        llvm::to_vector<4>(llvm::seq<int32_t>(0, dim)));
  } else {
    indicesAttr = rewriter.getI64VectorAttr(
        llvm::to_vector<4>(llvm::seq<int64_t>(0, dim)));
  }
  Value indices = rewriter.create<arith::ConstantOp>(loc, indicesAttr);
  // Add in an offset if requested.
  if (off) {
    Value o = getValueOrCreateCastToIndexLike(rewriter, loc, idxType, *off);
    Value ov = rewriter.create<vector::SplatOp>(loc, indices.getType(), o);
    indices = rewriter.create<arith::AddIOp>(loc, ov, indices);
  }
  // Construct the vector comparison.
  Value bound = getValueOrCreateCastToIndexLike(rewriter, loc, idxType, b);
  Value bounds =
      rewriter.create<vector::SplatOp>(loc, indices.getType(), bound);
  return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, indices,
                                        bounds);
}

template <typename ConcreteOp>
struct MaterializeTransferMask : public OpRewritePattern<ConcreteOp> {
public:
  explicit MaterializeTransferMask(MLIRContext *context, bool enableIndexOpt,
                                   PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<ConcreteOp>(context, benefit),
        force32BitVectorIndices(enableIndexOpt) {}

  LogicalResult matchAndRewrite(ConcreteOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (!xferOp.hasOutOfBoundsDim())
      return failure();

    if (xferOp.getVectorType().getRank() > 1 || xferOp.getIndices().empty())
      return failure();

    Location loc = xferOp->getLoc();
    VectorType vtp = xferOp.getVectorType();

    // Create the in-bounds mask with all elements between [0 .. dim - offset)
    // set and [dim - offset .. vector_length) unset.
    //
    // TODO: when the leaf transfer rank is k > 1, we need the last `k`
    //       dimensions here.
    unsigned lastIndex = llvm::size(xferOp.getIndices()) - 1;
    Value off = xferOp.getIndices()[lastIndex];
    Value dim =
        vector::createOrFoldDimOp(rewriter, loc, xferOp.getSource(), lastIndex);
    Value b = rewriter.create<arith::SubIOp>(loc, dim.getType(), dim, off);
    Value mask = rewriter.create<vector::CreateMaskOp>(
        loc,
        VectorType::get(vtp.getShape(), rewriter.getI1Type(),
                        vtp.getNumScalableDims()),
        b);
    if (xferOp.getMask()) {
      // Intersect the in-bounds with the mask specified as an op parameter.
      mask = rewriter.create<arith::AndIOp>(loc, mask, xferOp.getMask());
    }

    rewriter.updateRootInPlace(xferOp, [&]() {
      xferOp.getMaskMutable().assign(mask);
      xferOp.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));
    });

    return success();
  }

private:
  const bool force32BitVectorIndices;
};

/// Conversion pattern for a `vector.create_mask` (0-D and 1-D only).
class VectorCreateMaskOpConversion
    : public OpRewritePattern<vector::CreateMaskOp> {
public:
  explicit VectorCreateMaskOpConversion(MLIRContext *context,
                                        bool enableIndexOpt,
                                        PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<vector::CreateMaskOp>(context, benefit),
        force32BitVectorIndices(enableIndexOpt) {}

  LogicalResult matchAndRewrite(vector::CreateMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.getType();
    if (dstType.cast<VectorType>().isScalable())
      return failure();
    int64_t rank = dstType.getRank();
    if (rank > 1)
      return failure();
    rewriter.replaceOp(
        op, buildVectorComparison(rewriter, op, force32BitVectorIndices,
                                  rank == 0 ? 0 : dstType.getDimSize(0),
                                  op.getOperand(0)));
    return success();
  }

private:
  const bool force32BitVectorIndices;
};

// Drop inner most contiguous unit dimensions from transfer_read operand.
class DropInnerMostUnitDims : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (readOp.getTransferRank() == 0)
      return failure();

    // TODO: support mask.
    if (readOp.getMask())
      return failure();

    auto srcType = readOp.getSource().getType().dyn_cast<MemRefType>();
    if (!srcType || !srcType.hasStaticShape())
      return failure();

    if (!readOp.getPermutationMap().isMinorIdentity())
      return failure();

    auto targetType = readOp.getVectorType();
    if (targetType.getRank() <= 1)
      return failure();

    SmallVector<int64_t> srcStrides;
    int64_t srcOffset;
    if (failed(getStridesAndOffset(srcType, srcStrides, srcOffset)))
      return failure();

    size_t dimsToDrop = 0;
    for (size_t i = 1; i < srcStrides.size(); ++i) {
      int dim = srcType.getRank() - i - 1;
      if (srcStrides[dim] == 1) {
        dimsToDrop++;
      } else {
        break;
      }
    }
    if (dimsToDrop == 0)
      return failure();

    auto resultTargetVecType =
        VectorType::get(targetType.getShape().drop_back(dimsToDrop),
                        targetType.getElementType());

    MemRefType resultMemrefType;
    MemRefLayoutAttrInterface layout = srcType.getLayout();
    if (layout.isa<AffineMapAttr>() && layout.isIdentity()) {
      resultMemrefType = MemRefType::get(
          srcType.getShape().drop_back(dimsToDrop), srcType.getElementType(),
          nullptr, srcType.getMemorySpace());
    } else {
      MemRefLayoutAttrInterface updatedLayout;
      if (auto strided = layout.dyn_cast<StridedLayoutAttr>()) {
        auto strides =
            llvm::to_vector(strided.getStrides().drop_back(dimsToDrop));
        updatedLayout = StridedLayoutAttr::get(strided.getContext(),
                                               strided.getOffset(), strides);
      } else {
        AffineMap map = srcType.getLayout().getAffineMap();
        int numSymbols = map.getNumSymbols();
        for (size_t i = 0; i < dimsToDrop; ++i) {
          int dim = srcType.getRank() - i - 1;
          map = map.replace(rewriter.getAffineDimExpr(dim),
                            rewriter.getAffineConstantExpr(0),
                            map.getNumDims() - 1, numSymbols);
        }
      }
      resultMemrefType = MemRefType::get(
          srcType.getShape().drop_back(dimsToDrop), srcType.getElementType(),
          updatedLayout, srcType.getMemorySpace());
    }

    auto loc = readOp.getLoc();
    SmallVector<int64_t> offsets(srcType.getRank(), 0);
    SmallVector<int64_t> strides(srcType.getRank(), 1);

    ArrayAttr inBoundsAttr =
        readOp.getInBounds()
            ? rewriter.getArrayAttr(
                  readOp.getInBoundsAttr().getValue().drop_back(dimsToDrop))
            : ArrayAttr();
    Value rankedReducedView = rewriter.create<memref::SubViewOp>(
        loc, resultMemrefType, readOp.getSource(), offsets, srcType.getShape(),
        strides);
    auto permMap = getTransferMinorIdentityMap(
        rankedReducedView.getType().cast<ShapedType>(), resultTargetVecType);
    Value result = rewriter.create<vector::TransferReadOp>(
        loc, resultTargetVecType, rankedReducedView,
        readOp.getIndices().drop_back(dimsToDrop), AffineMapAttr::get(permMap),
        readOp.getPadding(),
        // TODO: support mask.
        /*mask=*/Value(), inBoundsAttr);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(readOp, targetType,
                                                     result);
    return success();
  }
};

namespace {

/// This function checks to see if the vector combining kind
/// is consistent with the integer or float element type.
static bool isValidKind(bool isInt, vector::CombiningKind kind) {
  using vector::CombiningKind;
  enum class KindType { FLOAT, INT, INVALID };
  KindType type{KindType::INVALID};
  switch (kind) {
  case CombiningKind::MINF:
  case CombiningKind::MAXF:
    type = KindType::FLOAT;
    break;
  case CombiningKind::MINUI:
  case CombiningKind::MINSI:
  case CombiningKind::MAXUI:
  case CombiningKind::MAXSI:
  case CombiningKind::AND:
  case CombiningKind::OR:
  case CombiningKind::XOR:
    type = KindType::INT;
    break;
  case CombiningKind::ADD:
  case CombiningKind::MUL:
    type = isInt ? KindType::INT : KindType::FLOAT;
    break;
  }
  bool isValidIntKind = (type == KindType::INT) && isInt;
  bool isValidFloatKind = (type == KindType::FLOAT) && (!isInt);
  return (isValidIntKind || isValidFloatKind);
}

/// This function constructs the appropriate integer or float
/// operation given the vector combining kind and operands. The
/// supported int operations are : add, mul, min (signed/unsigned),
/// max(signed/unsigned), and, or, xor. The supported float
/// operations are : add, mul, min and max.
static Value genOperator(Location loc, Value x, Value y,
                         vector::CombiningKind kind,
                         PatternRewriter &rewriter) {
  using vector::CombiningKind;

  auto elType = x.getType().cast<VectorType>().getElementType();
  bool isInt = elType.isIntOrIndex();

  Value combinedResult{nullptr};
  switch (kind) {
  case CombiningKind::ADD:
    if (isInt)
      combinedResult = rewriter.create<arith::AddIOp>(loc, x, y);
    else
      combinedResult = rewriter.create<arith::AddFOp>(loc, x, y);
    break;
  case CombiningKind::MUL:
    if (isInt)
      combinedResult = rewriter.create<arith::MulIOp>(loc, x, y);
    else
      combinedResult = rewriter.create<arith::MulFOp>(loc, x, y);
    break;
  case CombiningKind::MINUI:
    combinedResult = rewriter.create<arith::MinUIOp>(loc, x, y);
    break;
  case CombiningKind::MINSI:
    combinedResult = rewriter.create<arith::MinSIOp>(loc, x, y);
    break;
  case CombiningKind::MAXUI:
    combinedResult = rewriter.create<arith::MaxUIOp>(loc, x, y);
    break;
  case CombiningKind::MAXSI:
    combinedResult = rewriter.create<arith::MaxSIOp>(loc, x, y);
    break;
  case CombiningKind::AND:
    combinedResult = rewriter.create<arith::AndIOp>(loc, x, y);
    break;
  case CombiningKind::OR:
    combinedResult = rewriter.create<arith::OrIOp>(loc, x, y);
    break;
  case CombiningKind::XOR:
    combinedResult = rewriter.create<arith::XOrIOp>(loc, x, y);
    break;
  case CombiningKind::MINF:
    combinedResult = rewriter.create<arith::MinFOp>(loc, x, y);
    break;
  case CombiningKind::MAXF:
    combinedResult = rewriter.create<arith::MaxFOp>(loc, x, y);
    break;
  }
  return combinedResult;
}

/// Convert vector.scan op into arith ops and
/// vector.insert_strided_slice/extract_strided_slice
///
/// Ex:
/// ```
///   %0:2 = vector.scan <add>, %arg0, %arg1 {inclusive = true, reduction_dim =
///   1} :
///     (vector<2x3xi32>, vector<2xi32>) to (vector<2x3xi32>, vector<2xi32>)
/// ```
/// Gets converted to:
/// ```
///   %cst = arith.constant dense<0> : vector<2x3xi32>
///   %0 = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [2, 1],
///   strides = [1, 1]} : vector<2x3xi32> to vector<2x1xi32> %1 =
///   vector.insert_strided_slice %0, %cst {offsets = [0, 0], strides = [1, 1]}
///   : vector<2x1xi32> into vector<2x3xi32> %2 = vector.extract_strided_slice
///   %arg0 {offsets = [0, 1], sizes = [2, 1], strides = [1, 1]} :
///   vector<2x3xi32> to vector<2x1xi32> %3 = arith.muli %0, %2 :
///   vector<2x1xi32> %4 = vector.insert_strided_slice %3, %1 {offsets = [0, 1],
///   strides = [1, 1]} : vector<2x1xi32> into vector<2x3xi32> %5 =
///   vector.extract_strided_slice %arg0 {offsets = [0, 2], sizes = [2, 1],
///   strides = [1, 1]} : vector<2x3xi32> to vector<2x1xi32> %6 = arith.muli %3,
///   %5 : vector<2x1xi32> %7 = vector.insert_strided_slice %6, %4 {offsets =
///   [0, 2], strides = [1, 1]} : vector<2x1xi32> into vector<2x3xi32> %8 =
///   vector.shape_cast %6 : vector<2x1xi32> to vector<2xi32> return %7, %8 :
///   vector<2x3xi32>, vector<2xi32>
/// ```
struct ScanToArithOps : public OpRewritePattern<vector::ScanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ScanOp scanOp,
                                PatternRewriter &rewriter) const override {
    auto loc = scanOp.getLoc();
    VectorType destType = scanOp.getDestType();
    ArrayRef<int64_t> destShape = destType.getShape();
    auto elType = destType.getElementType();
    bool isInt = elType.isIntOrIndex();
    if (!isValidKind(isInt, scanOp.getKind()))
      return failure();

    VectorType resType = VectorType::get(destShape, elType);
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resType, rewriter.getZeroAttr(resType));
    int64_t reductionDim = scanOp.getReductionDim();
    bool inclusive = scanOp.getInclusive();
    int64_t destRank = destType.getRank();
    VectorType initialValueType = scanOp.getInitialValueType();
    int64_t initialValueRank = initialValueType.getRank();

    SmallVector<int64_t> reductionShape(destShape.begin(), destShape.end());
    reductionShape[reductionDim] = 1;
    VectorType reductionType = VectorType::get(reductionShape, elType);
    SmallVector<int64_t> offsets(destRank, 0);
    SmallVector<int64_t> strides(destRank, 1);
    SmallVector<int64_t> sizes(destShape.begin(), destShape.end());
    sizes[reductionDim] = 1;
    ArrayAttr scanSizes = rewriter.getI64ArrayAttr(sizes);
    ArrayAttr scanStrides = rewriter.getI64ArrayAttr(strides);

    Value lastOutput, lastInput;
    for (int i = 0; i < destShape[reductionDim]; i++) {
      offsets[reductionDim] = i;
      ArrayAttr scanOffsets = rewriter.getI64ArrayAttr(offsets);
      Value input = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, reductionType, scanOp.getSource(), scanOffsets, scanSizes,
          scanStrides);
      Value output;
      if (i == 0) {
        if (inclusive) {
          output = input;
        } else {
          if (initialValueRank == 0) {
            // ShapeCastOp cannot handle 0-D vectors
            output = rewriter.create<vector::BroadcastOp>(
                loc, input.getType(), scanOp.getInitialValue());
          } else {
            output = rewriter.create<vector::ShapeCastOp>(
                loc, input.getType(), scanOp.getInitialValue());
          }
        }
      } else {
        Value y = inclusive ? input : lastInput;
        output = genOperator(loc, lastOutput, y, scanOp.getKind(), rewriter);
        assert(output != nullptr);
      }
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, output, result, offsets, strides);
      lastOutput = output;
      lastInput = input;
    }

    Value reduction;
    if (initialValueRank == 0) {
      Value v = rewriter.create<vector::ExtractOp>(loc, lastOutput, 0);
      reduction =
          rewriter.create<vector::BroadcastOp>(loc, initialValueType, v);
    } else {
      reduction = rewriter.create<vector::ShapeCastOp>(loc, initialValueType,
                                                       lastOutput);
    }

    rewriter.replaceOp(scanOp, {result, reduction});
    return success();
  }
};

} // namespace

void mlir::vector::populateVectorMaskMaterializationPatterns(
    RewritePatternSet &patterns, bool force32BitVectorIndices,
    PatternBenefit benefit) {
  patterns.add<VectorCreateMaskOpConversion,
               MaterializeTransferMask<vector::TransferReadOp>,
               MaterializeTransferMask<vector::TransferWriteOp>>(
      patterns.getContext(), force32BitVectorIndices, benefit);
}

void mlir::vector::populateShapeCastFoldingPatterns(RewritePatternSet &patterns,
                                                    PatternBenefit benefit) {
  patterns.add<ShapeCastOpFolder>(patterns.getContext(), benefit);
}

void mlir::vector::populateBubbleVectorBitCastOpPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<BubbleDownVectorBitCastForExtract,
               BubbleDownBitCastForStridedSliceExtract,
               BubbleUpBitCastForStridedSliceInsert>(patterns.getContext(),
                                                     benefit);
}

void mlir::vector::populateVectorBroadcastLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<BroadcastOpLowering>(patterns.getContext(), benefit);
}

void mlir::vector::populateVectorMaskOpLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<CreateMaskOpLowering, ConstantMaskOpLowering>(
      patterns.getContext(), benefit);
}

void mlir::vector::populateVectorShapeCastLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ShapeCastOp2DDownCastRewritePattern,
               ShapeCastOp2DUpCastRewritePattern, ShapeCastOpRewritePattern>(
      patterns.getContext(), benefit);
}

void mlir::vector::populateVectorContractLoweringPatterns(
    RewritePatternSet &patterns, VectorTransformsOptions options,
    PatternBenefit benefit) {
  patterns.add<OuterProductOpLowering>(patterns.getContext(), benefit);
  patterns.add<ContractionOpLowering, ContractionOpToMatmulOpLowering,
               ContractionOpToOuterProductOpLowering>(
      options, patterns.getContext(), benefit);
}

void mlir::vector::populateVectorTransposeLoweringPatterns(
    RewritePatternSet &patterns, VectorTransformsOptions options,
    PatternBenefit benefit) {
  patterns.add<TransposeOpLowering, TransposeOp2DToShuffleLowering>(
      options, patterns.getContext(), benefit);
}

void mlir::vector::populateVectorReductionToContractPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<MultiReduceToContract, CombineContractBroadcast,
               CombineContractABTranspose, CombineContractResultTranspose,
               ReorderCastOpsOnBroadcast, ReorderElementwiseOpsOnTranspose>(
      patterns.getContext(), benefit);
}

void mlir::vector::
    populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
        RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<DropInnerMostUnitDims>(patterns.getContext(), benefit);
}

void mlir::vector::populateVectorTransferLoweringPatterns(
    RewritePatternSet &patterns, std::optional<unsigned> maxTransferRank,
    PatternBenefit benefit) {
  patterns.add<TransferReadToVectorLoadLowering,
               TransferWriteToVectorStoreLowering>(patterns.getContext(),
                                                   maxTransferRank, benefit);
  patterns
      .add<VectorLoadToMemrefLoadLowering, VectorStoreToMemrefStoreLowering>(
          patterns.getContext(), benefit);
}

void mlir::vector::populateVectorScanLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ScanToArithOps>(patterns.getContext(), benefit);
}

//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/VectorTransformsEnums.cpp.inc"
