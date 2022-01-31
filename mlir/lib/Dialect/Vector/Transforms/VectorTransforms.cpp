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

#include <type_traits>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
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
static Optional<int64_t> getResultIndex(AffineMap map, int64_t index) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
    int64_t idx = map.getDimPosition(i);
    if (idx == index)
      return i;
  }
  return None;
}

// Helper to construct iterator types with one index removed.
static SmallVector<Attribute, 4> adjustIter(ArrayAttr iteratorTypes,
                                            int64_t index) {
  SmallVector<Attribute, 4> results;
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
  SmallVector<AffineExpr, 4> results;
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
static SmallVector<IntType, 4> extractVector(ArrayAttr arrayAttr) {
  return llvm::to_vector<4>(llvm::map_range(
      arrayAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
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
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                PatternRewriter &rewriter) const override {
    // Check if 'shapeCastOp' has vector source/result type.
    auto sourceVectorType =
        shapeCastOp.source().getType().dyn_cast_or_null<VectorType>();
    auto resultVectorType =
        shapeCastOp.result().getType().dyn_cast_or_null<VectorType>();
    if (!sourceVectorType || !resultVectorType)
      return failure();

    // Check if shape cast op source operand is also a shape cast op.
    auto sourceShapeCastOp = dyn_cast_or_null<vector::ShapeCastOp>(
        shapeCastOp.source().getDefiningOp());
    if (!sourceShapeCastOp)
      return failure();
    auto operandSourceVectorType =
        sourceShapeCastOp.source().getType().cast<VectorType>();
    auto operandResultVectorType = sourceShapeCastOp.getType();

    // Check if shape cast operations invert each other.
    if (operandSourceVectorType != resultVectorType ||
        operandResultVectorType != sourceVectorType)
      return failure();

    rewriter.replaceOp(shapeCastOp, sourceShapeCastOp.source());
    return success();
  }
};

/// Progressive lowering of BroadcastOp.
class BroadcastOpLowering : public OpRewritePattern<vector::BroadcastOp> {
public:
  using OpRewritePattern<vector::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    VectorType dstType = op.getVectorType();
    VectorType srcType = op.getSourceType().dyn_cast<VectorType>();
    Type eltType = dstType.getElementType();

    // Scalar to any vector can use splat.
    if (!srcType) {
      rewriter.replaceOpWithNewOp<SplatOp>(op, dstType, op.source());
      return success();
    }

    // Determine rank of source and destination.
    int64_t srcRank = srcType.getRank();
    int64_t dstRank = dstType.getRank();

    // Stretching scalar inside vector (e.g. vector<1xf32>) can use splat.
    if (srcRank <= 1 && dstRank == 1) {
      Value ext;
      if (srcRank == 0)
        ext = rewriter.create<vector::ExtractElementOp>(loc, op.source());
      else
        ext = rewriter.create<vector::ExtractOp>(loc, op.source(), 0);
      rewriter.replaceOpWithNewOp<SplatOp>(op, dstType, ext);
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
          rewriter.create<vector::BroadcastOp>(loc, resType, op.source());
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
      rewriter.replaceOp(op, op.source());
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
      Value ext = rewriter.create<vector::ExtractOp>(loc, op.source(), 0);
      Value bcst = rewriter.create<vector::BroadcastOp>(loc, resType, ext);
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d)
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
    } else {
      // Stetch not at start.
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d) {
        Value ext = rewriter.create<vector::ExtractOp>(loc, op.source(), d);
        Value bcst = rewriter.create<vector::BroadcastOp>(loc, resType, ext);
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

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
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  TransposeOpLowering(vector::VectorTransformsOptions vectorTransformOptions,
                      MLIRContext *context)
      : OpRewritePattern<vector::TransposeOp>(context),
        vectorTransformOptions(vectorTransformOptions) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType resType = op.getResultType();

    // Set up convenience transposition table.
    SmallVector<int64_t, 4> transp;
    for (auto attr : op.transp())
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
          rewriter.create<vector::ShapeCastOp>(loc, flattenedType, op.vector());
      auto rows = rewriter.getI32IntegerAttr(resType.getShape()[0]);
      auto columns = rewriter.getI32IntegerAttr(resType.getShape()[1]);
      Value trans = rewriter.create<vector::FlatTransposeOp>(
          loc, flattenedType, matrix, rows, columns);
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resType, trans);
      return success();
    }

    // Generate fully unrolled extract/insert ops.
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resType, rewriter.getZeroAttr(resType));
    SmallVector<int64_t, 4> lhs(transp.size(), 0);
    SmallVector<int64_t, 4> rhs(transp.size(), 0);
    rewriter.replaceOp(op, expandIndices(loc, resType, 0, transp, lhs, rhs,
                                         op.vector(), result, rewriter));
    return success();
  }

private:
  // Builds the indices arrays for the lhs and rhs. Generates the extract/insert
  // operation when al ranks are exhausted.
  Value expandIndices(Location loc, VectorType resType, int64_t pos,
                      SmallVector<int64_t, 4> &transp,
                      SmallVector<int64_t, 4> &lhs,
                      SmallVector<int64_t, 4> &rhs, Value input, Value result,
                      PatternRewriter &rewriter) const {
    if (pos >= resType.getRank()) {
      auto ridx = rewriter.getI64ArrayAttr(rhs);
      auto lidx = rewriter.getI64ArrayAttr(lhs);
      Type eltType = resType.getElementType();
      Value e = rewriter.create<vector::ExtractOp>(loc, eltType, input, ridx);
      return rewriter.create<vector::InsertOp>(loc, resType, e, result, lidx);
    }
    for (int64_t d = 0, e = resType.getDimSize(pos); d < e; ++d) {
      lhs[pos] = d;
      rhs[transp[pos]] = d;
      result = expandIndices(loc, resType, pos + 1, transp, lhs, rhs, input,
                             result, rewriter);
    }
    return result;
  }

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
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  TransposeOp2DToShuffleLowering(
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context)
      : OpRewritePattern<vector::TransposeOp>(context),
        vectorTransformOptions(vectorTransformOptions) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType srcType = op.getVectorType();
    if (srcType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "Not a 2D transpose");

    SmallVector<int64_t, 4> transp;
    for (auto attr : op.transp())
      transp.push_back(attr.cast<IntegerAttr>().getInt());
    if (transp[0] != 1 && transp[1] != 0)
      return rewriter.notifyMatchFailure(op, "Not a 2D transpose permutation");

    if (vectorTransformOptions.vectorTransposeLowering !=
        VectorTransposeLowering::Shuffle)
      return rewriter.notifyMatchFailure(op, "Options do not ask for Shuffle");

    int64_t m = srcType.getShape().front(), n = srcType.getShape().back();
    Value casted = rewriter.create<vector::ShapeCastOp>(
        loc, VectorType::get({m * n}, srcType.getElementType()), op.vector());
    SmallVector<int64_t> mask;
    mask.reserve(m * n);
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < m; ++i)
        mask.push_back(i * n + j);

    Value shuffled =
        rewriter.create<vector::ShuffleOp>(loc, casted, casted, mask);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getResultType(),
                                                     shuffled);

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
  using OpRewritePattern<vector::OuterProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::OuterProductOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType lhsType = op.getOperandVectorTypeLHS();
    VectorType rhsType = op.getOperandTypeRHS().dyn_cast<VectorType>();
    VectorType resType = op.getVectorType();
    Type eltType = resType.getElementType();
    bool isInt = eltType.isa<IntegerType, IndexType>();
    Value acc = (op.acc().empty()) ? nullptr : op.acc()[0];
    vector::CombiningKind kind = op.kind();

    if (!rhsType) {
      // Special case: AXPY operation.
      Value b = rewriter.create<vector::BroadcastOp>(loc, lhsType, op.rhs());
      Optional<Value> mult =
          isInt ? genMultI(loc, op.lhs(), b, acc, kind, rewriter)
                : genMultF(loc, op.lhs(), b, acc, kind, rewriter);
      if (!mult.hasValue())
        return failure();
      rewriter.replaceOp(op, mult.getValue());
      return success();
    }

    Value result = rewriter.create<arith::ConstantOp>(
        loc, resType, rewriter.getZeroAttr(resType));
    for (int64_t d = 0, e = resType.getDimSize(0); d < e; ++d) {
      auto pos = rewriter.getI64ArrayAttr(d);
      Value x = rewriter.create<vector::ExtractOp>(loc, eltType, op.lhs(), pos);
      Value a = rewriter.create<vector::BroadcastOp>(loc, rhsType, x);
      Value r = nullptr;
      if (acc)
        r = rewriter.create<vector::ExtractOp>(loc, rhsType, acc, pos);
      Optional<Value> m = isInt ? genMultI(loc, a, op.rhs(), r, kind, rewriter)
                                : genMultF(loc, a, op.rhs(), r, kind, rewriter);
      if (!m.hasValue())
        return failure();
      result = rewriter.create<vector::InsertOp>(loc, resType, m.getValue(),
                                                 result, pos);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static Optional<Value> genMultI(Location loc, Value x, Value y, Value acc,
                                  vector::CombiningKind kind,
                                  PatternRewriter &rewriter) {
    using vector::CombiningKind;

    auto mul = rewriter.create<arith::MulIOp>(loc, x, y);
    if (!acc)
      return Optional<Value>(mul);

    Value combinedResult;
    switch (kind) {
    case CombiningKind::ADD:
      combinedResult = rewriter.create<arith::AddIOp>(loc, mul, acc);
      break;
    case CombiningKind::MUL:
      combinedResult = rewriter.create<arith::MulIOp>(loc, mul, acc);
      break;
    case CombiningKind::MINUI:
      combinedResult = rewriter.create<arith::MinUIOp>(loc, mul, acc);
      break;
    case CombiningKind::MINSI:
      combinedResult = rewriter.create<arith::MinSIOp>(loc, mul, acc);
      break;
    case CombiningKind::MAXUI:
      combinedResult = rewriter.create<arith::MaxUIOp>(loc, mul, acc);
      break;
    case CombiningKind::MAXSI:
      combinedResult = rewriter.create<arith::MaxSIOp>(loc, mul, acc);
      break;
    case CombiningKind::AND:
      combinedResult = rewriter.create<arith::AndIOp>(loc, mul, acc);
      break;
    case CombiningKind::OR:
      combinedResult = rewriter.create<arith::OrIOp>(loc, mul, acc);
      break;
    case CombiningKind::XOR:
      combinedResult = rewriter.create<arith::XOrIOp>(loc, mul, acc);
      break;
    case CombiningKind::MINF: // Only valid for floating point types.
    case CombiningKind::MAXF: // Only valid for floating point types.
      return Optional<Value>();
    }
    return Optional<Value>(combinedResult);
  }

  static Optional<Value> genMultF(Location loc, Value x, Value y, Value acc,
                                  vector::CombiningKind kind,
                                  PatternRewriter &rewriter) {
    using vector::CombiningKind;

    // Special case for fused multiply-add.
    if (acc && kind == CombiningKind::ADD) {
      return Optional<Value>(rewriter.create<vector::FMAOp>(loc, x, y, acc));
    }

    auto mul = rewriter.create<arith::MulFOp>(loc, x, y);

    if (!acc)
      return Optional<Value>(mul);

    Value combinedResult;
    switch (kind) {
    case CombiningKind::MUL:
      combinedResult = rewriter.create<arith::MulFOp>(loc, mul, acc);
      break;
    case CombiningKind::MINF:
      combinedResult = rewriter.create<arith::MinFOp>(loc, mul, acc);
      break;
    case CombiningKind::MAXF:
      combinedResult = rewriter.create<arith::MaxFOp>(loc, mul, acc);
      break;
    case CombiningKind::ADD:   // Already handled this special case above.
    case CombiningKind::AND:   // Only valid for integer types.
    case CombiningKind::MINUI: // Only valid for integer types.
    case CombiningKind::MINSI: // Only valid for integer types.
    case CombiningKind::MAXUI: // Only valid for integer types.
    case CombiningKind::MAXSI: // Only valid for integer types.
    case CombiningKind::OR:    // Only valid for integer types.
    case CombiningKind::XOR:   // Only valid for integer types.
      return Optional<Value>();
    }
    return Optional<Value>(combinedResult);
  }
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
  using OpRewritePattern<vector::ConstantMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ConstantMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstType = op.getType();
    auto eltType = dstType.getElementType();
    auto dimSizes = op.mask_dim_sizes();
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

    int64_t trueDim = std::min(dstType.getDimSize(0),
                               dimSizes[0].cast<IntegerAttr>().getInt());

    if (rank == 1) {
      // Express constant 1-D case in explicit vector form:
      //   [T,..,T,F,..,F].
      SmallVector<bool, 4> values(dstType.getDimSize(0));
      for (int64_t d = 0; d < trueDim; d++)
        values[d] = true;
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, dstType, rewriter.getBoolVectorAttr(values));
      return success();
    }

    VectorType lowType =
        VectorType::get(dstType.getShape().drop_front(), eltType);
    SmallVector<int64_t, 4> newDimSizes;
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
  using OpRewritePattern<vector::CreateMaskOp>::OpRewritePattern;

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
      Value sel = rewriter.create<SelectOp>(loc, val, trueVal, falseVal);
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
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;

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
      Value vec = rewriter.create<vector::ExtractOp>(loc, op.source(), i);
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
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;

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
          loc, op.source(), /*offsets=*/i * mostMinorVectorSize,
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
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;

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
    SmallVector<int64_t, 4> srcIdx(srcRank);
    SmallVector<int64_t, 4> resIdx(resRank);
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    for (int64_t i = 0; i < numElts; i++) {
      if (i != 0) {
        incIdx(srcIdx, sourceVectorType, srcRank - 1);
        incIdx(resIdx, resultVectorType, resRank - 1);
      }
      Value e = rewriter.create<vector::ExtractOp>(loc, op.source(), srcIdx);
      result = rewriter.create<vector::InsertOp>(loc, e, result, resIdx);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static void incIdx(SmallVector<int64_t, 4> &idx, VectorType tp, int64_t r) {
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
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.kind() != vector::CombiningKind::ADD)
      return failure();
    Operation *mulOp = reduceOp.source().getDefiningOp();
    if (!mulOp || !isa<arith::MulIOp, arith::MulFOp>(mulOp))
      return failure();
    SmallVector<bool> reductionMask = reduceOp.getReductionMask();
    auto srcMap = rewriter.getMultiDimIdentityMap(reductionMask.size());
    SmallVector<AffineExpr> exprs;
    SmallVector<StringRef> iteratorTypes;
    for (const auto &isReduceDim : llvm::enumerate(reductionMask)) {
      if (!isReduceDim.value()) {
        iteratorTypes.push_back(getParallelIteratorTypeName());
        exprs.push_back(rewriter.getAffineDimExpr(isReduceDim.index()));
      } else {
        iteratorTypes.push_back(getReductionIteratorTypeName());
      }
    }
    auto dstMap = AffineMap::get(/*dimCount=*/reductionMask.size(),
                                 /*symCount=*/0, exprs, reduceOp.getContext());
    Value zero = rewriter.create<arith::ConstantOp>(
        reduceOp.getLoc(), reduceOp.getDestType(),
        rewriter.getZeroAttr(reduceOp.getDestType()));
    rewriter.replaceOpWithNewOp<mlir::vector::ContractionOp>(
        reduceOp, mulOp->getOperand(0), mulOp->getOperand(1), zero,
        rewriter.getAffineMapArrayAttr({srcMap, srcMap, dstMap}),
        rewriter.getStrArrayAttr(iteratorTypes));
    return success();
  }
};

/// Merge TransposeOp into ContractionOp user.
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
struct CombineContractTranspose
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap, 4> maps =
        llvm::to_vector<4>(contractOp.getIndexingMaps());
    Value lhs = contractOp.lhs();
    Value rhs = contractOp.rhs();
    size_t index = 0;
    bool changed = false;
    for (Value *operand : {&lhs, &rhs}) {
      AffineMap &map = maps[index++];
      auto transposeOp = operand->getDefiningOp<vector::TransposeOp>();
      if (!transposeOp)
        continue;
      SmallVector<int64_t> perm;
      transposeOp.getTransp(perm);
      AffineMap permutationMap = AffineMap::getPermutationMap(
          extractVector<unsigned>(transposeOp.transp()),
          contractOp.getContext());
      map = inversePermutation(permutationMap).compose(map);
      *operand = transposeOp.vector();
      changed = true;
    }
    if (!changed)
      return failure();
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, lhs, rhs, contractOp.acc(),
        rewriter.getAffineMapArrayAttr(maps), contractOp.iterator_types());
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
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap, 4> maps =
        llvm::to_vector<4>(contractOp.getIndexingMaps());
    Value lhs = contractOp.lhs();
    Value rhs = contractOp.rhs();
    size_t index = 0;
    bool changed = false;
    for (Value *operand : {&lhs, &rhs}) {
      AffineMap &map = maps[index++];
      auto broadcast = operand->getDefiningOp<vector::BroadcastOp>();
      if (!broadcast)
        continue;
      // contractionOp can only take vector as operands.
      auto srcType = broadcast.getSourceType().dyn_cast<VectorType>();
      if (!srcType || srcType.getRank() == broadcast.getVectorType().getRank())
        continue;
      int64_t rankDiff =
          broadcast.getVectorType().getRank() - srcType.getRank();
      bool innerDimBroadcast = false;
      SmallVector<AffineExpr> originalDims;
      for (const auto &dim : llvm::enumerate(srcType.getShape())) {
        if (dim.value() !=
            broadcast.getVectorType().getDimSize(rankDiff + dim.index())) {
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
      AffineMap broadcastMap =
          AffineMap::get(broadcast.getVectorType().getRank(), 0, originalDims,
                         contractOp.getContext());
      map = broadcastMap.compose(map);
      *operand = broadcast.source();
      changed = true;
    }
    if (!changed)
      return failure();
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, lhs, rhs, contractOp.acc(),
        rewriter.getAffineMapArrayAttr(maps), contractOp.iterator_types());
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
  // TODO: implement masks
  if (llvm::size(op.masks()) != 0)
    return failure();
  if (vectorTransformOptions.vectorContractLowering !=
      vector::VectorContractLowering::Matmul)
    return failure();
  if (failed(filter(op)))
    return failure();

  auto iteratorTypes = op.iterator_types().getValue();
  if (!isParallelIterator(iteratorTypes[0]) ||
      !isParallelIterator(iteratorTypes[1]) ||
      !isReductionIterator(iteratorTypes[2]))
    return failure();

  Type elementType = op.getLhsType().getElementType();
  if (!elementType.isIntOrFloat())
    return failure();

  // Perform lhs + rhs transpositions to conform to matmul row-major semantics.
  // Bail out if the contraction cannot be put in this form.
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  AffineExpr m, n, k;
  bindDims(rew.getContext(), m, n, k);
  // LHS must be A(m, k) or A(k, m).
  Value lhs = op.lhs();
  auto lhsMap = op.indexing_maps()[0].cast<AffineMapAttr>().getValue();
  if (lhsMap == AffineMap::get(3, 0, {k, m}, ctx))
    lhs = rew.create<vector::TransposeOp>(loc, lhs, ArrayRef<int64_t>{1, 0});
  else if (lhsMap != AffineMap::get(3, 0, {m, k}, ctx))
    return failure();

  // RHS must be B(k, n) or B(n, k).
  Value rhs = op.rhs();
  auto rhsMap = op.indexing_maps()[1].cast<AffineMapAttr>().getValue();
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
                      getElementTypeOrSelf(op.acc().getType())),
      mul);

  // ACC must be C(m, n) or C(n, m).
  auto accMap = op.indexing_maps()[2].cast<AffineMapAttr>().getValue();
  if (accMap == AffineMap::get(3, 0, {n, m}, ctx))
    mul = rew.create<vector::TransposeOp>(loc, mul, ArrayRef<int64_t>{1, 0});
  else if (accMap != AffineMap::get(3, 0, {m, n}, ctx))
    llvm_unreachable("invalid contraction semantics");

  Value res =
      elementType.isa<IntegerType>()
          ? static_cast<Value>(rew.create<arith::AddIOp>(loc, op.acc(), mul))
          : static_cast<Value>(rew.create<arith::AddFOp>(loc, op.acc(), mul));

  rew.replaceOp(op, res);
  return success();
}

namespace {
struct IteratorType {
  IteratorType(StringRef strRef) : strRef(strRef) {}
  bool isOfType(Attribute attr) const {
    auto sAttr = attr.dyn_cast<StringAttr>();
    return sAttr && sAttr.getValue() == strRef;
  }
  StringRef strRef;
};
struct Par : public IteratorType {
  Par() : IteratorType(getParallelIteratorTypeName()) {}
};
struct Red : public IteratorType {
  Red() : IteratorType(getReductionIteratorTypeName()) {}
};

/// Generate a vector implementation for matmat, matvec and tmatvec.
/// This unrolls outer-products along the reduction dimension.
struct UnrolledOuterProductGenerator
    : public StructuredGenerator<vector::ContractionOp> {

  UnrolledOuterProductGenerator(OpBuilder &builder, vector::ContractionOp op)
      : StructuredGenerator<vector::ContractionOp>(builder, op),
        kind(op.kind()), lhs(op.lhs()), rhs(op.rhs()), res(op.acc()),
        lhsType(op.getLhsType()) {}

  Value t(Value v) {
    static constexpr std::array<int64_t, 2> perm = {1, 0};
    return builder.create<vector::TransposeOp>(loc, v, perm);
  }

  Value outerProd(Value lhs, Value rhs, Value res, int reductionSize) {
    assert(reductionSize > 0);
    for (int64_t k = 0; k < reductionSize; ++k) {
      Value a = builder.create<vector::ExtractOp>(loc, lhs, k);
      Value b = builder.create<vector::ExtractOp>(loc, rhs, k);
      res = builder.create<vector::OuterProductOp>(loc, res.getType(), a, b,
                                                   res, kind);
    }
    return res;
  }

  /// Two outer parallel, one inner reduction (matmat flavor).
  FailureOr<Value> matmat() {
    if (!iters({Par(), Par(), Red()}))
      return failure();
    // Set up the parallel/reduction structure in the right form.
    AffineExpr m, n, k;
    bindDims(builder.getContext(), m, n, k);
    // Classical row-major matmul:  Just permute the lhs.
    if (layout({{m, k}, {k, n}, {m, n}}))
      return outerProd(t(lhs), rhs, res, lhsType.getDimSize(1));
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
    bindDims(builder.getContext(), m, k);

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

  //
  // One outer reduction, one inner parallel (tmatvec flavor)
  //
  FailureOr<Value> tmatvec() {
    if (!iters({Red(), Par()}))
      return failure();
    AffineExpr k, m;
    bindDims(builder.getContext(), k, m);

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
  Value lhs, rhs, res;
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
  // TODO: implement masks
  if (llvm::size(op.masks()) != 0)
    return failure();

  if (vectorTransformOptions.vectorContractLowering !=
      vector::VectorContractLowering::OuterProduct)
    return failure();

  if (failed(filter(op)))
    return failure();

  UnrolledOuterProductGenerator e(rewriter, op);
  FailureOr<Value> matmatRes = e.matmat();
  if (succeeded(matmatRes)) {
    rewriter.replaceOp(op, *matmatRes);
    return success();
  }
  FailureOr<Value> matvecRes = e.matvec();
  if (succeeded(matvecRes)) {
    rewriter.replaceOp(op, *matvecRes);
    return success();
  }
  FailureOr<Value> tmatvecRes = e.tmatvec();
  if (succeeded(tmatvecRes)) {
    rewriter.replaceOp(op, *tmatvecRes);
    return success();
  }

  return failure();
}

LogicalResult
ContractionOpToDotLowering::matchAndRewrite(vector::ContractionOp op,
                                            PatternRewriter &rewriter) const {
  // TODO: implement masks
  if (llvm::size(op.masks()) != 0)
    return failure();

  if (failed(filter(op)))
    return failure();

  if (vectorTransformOptions.vectorContractLowering !=
      vector::VectorContractLowering::Dot)
    return failure();

  auto iteratorTypes = op.iterator_types().getValue();
  static constexpr std::array<int64_t, 2> perm = {1, 0};
  Location loc = op.getLoc();
  Value lhs = op.lhs(), rhs = op.rhs();

  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n, k;
  bindDims(rewriter.getContext(), m, n, k);
  SmallVector<AffineMap, 4> maps = op.getIndexingMaps();
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
          op.getLoc(), dstType.getElementType(), rewriter.getStringAttr("add"),
          m, ValueRange{});

      SmallVector<int64_t, 2> pos = rank == 1 ? SmallVector<int64_t, 2>{r}
                                              : SmallVector<int64_t, 2>{r, c};
      res = rewriter.create<vector::InsertOp>(op.getLoc(), reduced, res, pos);
    }
  }
  if (auto acc = op.acc())
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
  // TODO: implement masks.
  if (llvm::size(op.masks()) != 0)
    return failure();

  if (failed(filter(op)))
    return failure();

  // TODO: support mixed mode contract lowering.
  if (op.getLhsType().getElementType() !=
          getElementTypeOrSelf(op.getAccType()) ||
      op.getRhsType().getElementType() != getElementTypeOrSelf(op.getAccType()))
    return failure();

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

  // Find first batch dimension in LHS/RHS, and lower when found.
  std::vector<std::pair<int64_t, int64_t>> batchDimMap = op.getBatchDimMap();
  if (!batchDimMap.empty()) {
    int64_t lhsIndex = batchDimMap[0].first;
    int64_t rhsIndex = batchDimMap[0].second;
    rewriter.replaceOp(op, lowerParallel(op, lhsIndex, rhsIndex, rewriter));
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
      rewriter.replaceOp(
          op, lowerParallel(op, lhsIndex, /*rhsIndex=*/-1, rewriter));
      return success();
    }
  }

  // Find first free dimension in RHS, and lower when found.
  VectorType rhsType = op.getRhsType();
  for (int64_t rhsIndex = 0, e = rhsType.getRank(); rhsIndex < e; ++rhsIndex) {
    if (rhsContractingDimSet.count(rhsIndex) == 0) {
      rewriter.replaceOp(
          op, lowerParallel(op, /*lhsIndex=*/-1, rhsIndex, rewriter));
      return success();
    }
  }

  // Lower the first remaining reduction dimension.
  if (!contractingDimMap.empty()) {
    rewriter.replaceOp(op, lowerReduction(op, rewriter));
    return success();
  }

  return failure();
}

// Lower one parallel dimension.
// TODO: consider reusing existing contract unrolling
Value ContractionOpLowering::lowerParallel(vector::ContractionOp op,
                                           int64_t lhsIndex, int64_t rhsIndex,
                                           PatternRewriter &rewriter) const {
  VectorType lhsType = op.getLhsType();
  VectorType rhsType = op.getRhsType();
  VectorType resType = op.getResultType().cast<VectorType>();
  // Find the iterator type index and result index.
  SmallVector<AffineMap, 4> iMap = op.getIndexingMaps();
  int64_t iterIndex = -1;
  int64_t dimSize = -1;
  if (lhsIndex >= 0) {
    iterIndex = iMap[0].getDimPosition(lhsIndex);
    assert((rhsIndex < 0 || iterIndex == iMap[1].getDimPosition(rhsIndex)) &&
           "parallel index should be free in LHS or batch in LHS/RHS");
    dimSize = lhsType.getDimSize(lhsIndex);
  } else {
    assert(rhsIndex >= 0 && "missing parallel index");
    iterIndex = iMap[1].getDimPosition(rhsIndex);
    dimSize = rhsType.getDimSize(rhsIndex);
  }
  assert(iterIndex >= 0 && "parallel index not listed in operand mapping");
  Optional<int64_t> lookup = getResultIndex(iMap[2], iterIndex);
  assert(lookup.hasValue() && "parallel index not listed in reduction");
  int64_t resIndex = lookup.getValue();
  // Construct new iterator types and affine map array attribute.
  std::array<AffineMap, 3> lowIndexingMaps = {
      adjustMap(iMap[0], iterIndex, rewriter),
      adjustMap(iMap[1], iterIndex, rewriter),
      adjustMap(iMap[2], iterIndex, rewriter)};
  auto lowAffine = rewriter.getAffineMapArrayAttr(lowIndexingMaps);
  auto lowIter =
      rewriter.getArrayAttr(adjustIter(op.iterator_types(), iterIndex));
  // Unroll into a series of lower dimensional vector.contract ops.
  Location loc = op.getLoc();
  Value result = rewriter.create<arith::ConstantOp>(
      loc, resType, rewriter.getZeroAttr(resType));
  for (int64_t d = 0; d < dimSize; ++d) {
    auto lhs = reshapeLoad(loc, op.lhs(), lhsType, lhsIndex, d, rewriter);
    auto rhs = reshapeLoad(loc, op.rhs(), rhsType, rhsIndex, d, rewriter);
    auto acc = reshapeLoad(loc, op.acc(), resType, resIndex, d, rewriter);
    Value lowContract = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, acc, lowAffine, lowIter);
    result =
        reshapeStore(loc, lowContract, result, resType, resIndex, d, rewriter);
  }
  return result;
}

// Lower one reduction dimension.
Value ContractionOpLowering::lowerReduction(vector::ContractionOp op,
                                            PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  VectorType lhsType = op.getLhsType();
  VectorType rhsType = op.getRhsType();
  Type resType = op.getResultType();
  assert(!resType.isa<VectorType>());
  bool isInt = resType.isa<IntegerType>();
  // Use iterator index 0.
  int64_t iterIndex = 0;
  SmallVector<AffineMap, 4> iMap = op.getIndexingMaps();
  Optional<int64_t> lookupLhs = getResultIndex(iMap[0], iterIndex);
  Optional<int64_t> lookupRhs = getResultIndex(iMap[1], iterIndex);
  assert(lookupLhs.hasValue() && "missing LHS parallel index");
  assert(lookupRhs.hasValue() && "missing RHS parallel index");
  int64_t lhsIndex = lookupLhs.getValue();
  int64_t rhsIndex = lookupRhs.getValue();
  int64_t dimSize = lhsType.getDimSize(lhsIndex);
  assert(dimSize == rhsType.getDimSize(rhsIndex) && "corrupt shape");
  // Base case.
  if (lhsType.getRank() == 1) {
    assert(rhsType.getRank() == 1 && "corrupt contraction");
    Value m = createMul(loc, op.lhs(), op.rhs(), isInt, rewriter);
    StringAttr kind = rewriter.getStringAttr("add");
    Value res = rewriter.create<vector::ReductionOp>(loc, resType, kind, m,
                                                     ValueRange{});
    if (auto acc = op.acc())
      res = createAdd(op.getLoc(), res, acc, isInt, rewriter);
    return res;
  }
  // Construct new iterator types and affine map array attribute.
  std::array<AffineMap, 3> lowIndexingMaps = {
      adjustMap(iMap[0], iterIndex, rewriter),
      adjustMap(iMap[1], iterIndex, rewriter),
      adjustMap(iMap[2], iterIndex, rewriter)};
  auto lowAffine = rewriter.getAffineMapArrayAttr(lowIndexingMaps);
  auto lowIter =
      rewriter.getArrayAttr(adjustIter(op.iterator_types(), iterIndex));
  // Unroll into a series of lower dimensional vector.contract ops.
  // By feeding the initial accumulator into the first contraction,
  // and the result of each contraction into the next, eventually
  // the sum of all reductions is computed.
  Value result = op.acc();
  for (int64_t d = 0; d < dimSize; ++d) {
    auto lhs = reshapeLoad(loc, op.lhs(), lhsType, lhsIndex, d, rewriter);
    auto rhs = reshapeLoad(loc, op.rhs(), rhsType, rhsIndex, d, rewriter);
    result = rewriter.create<vector::ContractionOp>(loc, lhs, rhs, result,
                                                    lowAffine, lowIter);
  }
  return result;
}

} // namespace mlir

Optional<mlir::vector::DistributeOps> mlir::vector::distributPointwiseVectorOp(
    OpBuilder &builder, Operation *op, ArrayRef<Value> ids,
    ArrayRef<int64_t> multiplicity, const AffineMap &map) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  if (op->getNumResults() != 1)
    return {};
  Value result = op->getResult(0);
  VectorType type = op->getResult(0).getType().dyn_cast<VectorType>();
  if (!type || map.getNumResults() != multiplicity.size())
    return {};
  // For each dimension being distributed check that the size is a multiple of
  // the multiplicity. To handle more sizes we would need to support masking.
  unsigned multiplictyCount = 0;
  for (auto exp : map.getResults()) {
    auto affinExp = exp.dyn_cast<AffineDimExpr>();
    if (!affinExp || affinExp.getPosition() >= type.getRank() ||
        type.getDimSize(affinExp.getPosition()) %
                multiplicity[multiplictyCount++] !=
            0)
      return {};
  }
  DistributeOps ops;
  ops.extract =
      builder.create<vector::ExtractMapOp>(loc, result, ids, multiplicity, map);
  ops.insert =
      builder.create<vector::InsertMapOp>(loc, ops.extract, result, ids);
  return ops;
}

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
                                   llvm::Optional<unsigned> maxRank)
      : OpRewritePattern<vector::TransferReadOp>(context),
        maxTransferRank(maxRank) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    if (maxTransferRank && read.getVectorType().getRank() > *maxTransferRank)
      return failure();

    SmallVector<unsigned, 4> broadcastedDims;
    // Permutations are handled by VectorToSCF or
    // populateVectorTransferPermutationMapLoweringPatterns.
    // We let the 0-d corner case pass-through as it is supported.
    if (!read.permutation_map().isMinorIdentityWithBroadcasting(
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
    SmallVector<int64_t, 4> unbroadcastedVectorShape(vectorShape.begin(),
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
    if (read.mask()) {
      Value fill = rewriter.create<SplatOp>(
          read.getLoc(), unbroadcastedVectorType, read.padding());
      loadOp = rewriter.create<vector::MaskedLoadOp>(
          read.getLoc(), unbroadcastedVectorType, read.source(), read.indices(),
          read.mask(), fill);
    } else {
      loadOp = rewriter.create<vector::LoadOp>(read.getLoc(),
                                               unbroadcastedVectorType,
                                               read.source(), read.indices());
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

  llvm::Optional<unsigned> maxTransferRank;
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
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = loadOp.getVectorType();
    if (vecType.getNumElements() != 1)
      return failure();
    auto memrefLoad = rewriter.create<memref::LoadOp>(
        loadOp.getLoc(), loadOp.base(), loadOp.indices());
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(loadOp, vecType,
                                                     memrefLoad);
    return success();
  }
};

/// Replace a 0-d vector.store with a vector.extractelement + memref.store.
struct VectorStoreToMemrefStoreLowering
    : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = storeOp.getVectorType();
    if (vecType.getNumElements() != 1)
      return failure();
    Value extracted;
    if (vecType.getRank() == 0) {
      // TODO: Unifiy once ExtractOp supports 0-d vectors.
      extracted = rewriter.create<vector::ExtractElementOp>(
          storeOp.getLoc(), storeOp.valueToStore());
    } else {
      SmallVector<int64_t> indices(vecType.getRank(), 0);
      extracted = rewriter.create<vector::ExtractOp>(
          storeOp.getLoc(), storeOp.valueToStore(), indices);
    }

    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        storeOp, extracted, storeOp.base(), storeOp.indices());
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
                                     llvm::Optional<unsigned> maxRank)
      : OpRewritePattern<vector::TransferWriteOp>(context),
        maxTransferRank(maxRank) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    if (maxTransferRank && write.getVectorType().getRank() > *maxTransferRank)
      return failure();

    // Permutations are handled by VectorToSCF or
    // populateVectorTransferPermutationMapLoweringPatterns.
    if ( // pass-through for the 0-d corner case.
        !write.permutation_map().isMinorIdentity())
      return failure();

    auto memRefType = write.getShapedType().dyn_cast<MemRefType>();
    if (!memRefType)
      return failure();

    // Non-unit strides are handled by VectorToSCF.
    if (!vector::isLastMemrefDimUnitStride(memRefType))
      return failure();

    // `vector.store` supports vector types as memref's elements only when the
    // type of the vector value being written is the same as the element type.
    auto memrefElTy = memRefType.getElementType();
    if (memrefElTy.isa<VectorType>() && memrefElTy != write.getVectorType())
      return failure();

    // Otherwise, element types of the memref and the vector must match.
    if (!memrefElTy.isa<VectorType>() &&
        memrefElTy != write.getVectorType().getElementType())
      return failure();

    // Out-of-bounds dims are handled by MaterializeTransferMask.
    if (write.hasOutOfBoundsDim())
      return failure();
    if (write.mask()) {
      rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
          write, write.source(), write.indices(), write.mask(), write.vector());
    } else {
      rewriter.replaceOpWithNewOp<vector::StoreOp>(
          write, write.vector(), write.source(), write.indices());
    }
    return success();
  }

  llvm::Optional<unsigned> maxTransferRank;
};

// Returns the values in `arrayAttr` as an integer vector.
static SmallVector<int64_t, 4> getIntValueVector(ArrayAttr arrayAttr) {
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
    if (extractOp.getVectorType().getRank() != 1)
      return failure();

    auto castOp = extractOp.vector().getDefiningOp<vector::BitCastOp>();
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

    uint64_t index = getFirstIntValue(extractOp.position());

    // Get the single scalar (as a vector) in the source value that packs the
    // desired scalar. E.g. extract vector<1xf32> from vector<4xf32>
    VectorType oneScalarType =
        VectorType::get({1}, castSrcType.getElementType());
    Value packedValue = rewriter.create<vector::ExtractOp>(
        extractOp.getLoc(), oneScalarType, castOp.source(),
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
    auto castOp = extractOp.vector().getDefiningOp<vector::BitCastOp>();
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
    if (llvm::any_of(extractOp.strides().getAsValueRange<IntegerAttr>(),
                     [](const APInt &val) { return !val.isOneValue(); }))
      return failure();

    unsigned rank = extractOp.getVectorType().getRank();
    assert(castDstLastDim % castSrcLastDim == 0);
    int64_t expandRatio = castDstLastDim / castSrcLastDim;

    // If we have a less number of offsets than the rank, then implicitly we
    // are selecting the full range for the last bitcasted dimension; other
    // dimensions aren't affected. Otherwise, we need to scale down the last
    // dimension's offset given we are extracting from less elements now.
    ArrayAttr newOffsets = extractOp.offsets();
    if (newOffsets.size() == rank) {
      SmallVector<int64_t, 4> offsets = getIntValueVector(newOffsets);
      if (offsets.back() % expandRatio != 0)
        return failure();
      offsets.back() = offsets.back() / expandRatio;
      newOffsets = rewriter.getI64ArrayAttr(offsets);
    }

    // Similarly for sizes.
    ArrayAttr newSizes = extractOp.sizes();
    if (newSizes.size() == rank) {
      SmallVector<int64_t, 4> sizes = getIntValueVector(newSizes);
      if (sizes.back() % expandRatio != 0)
        return failure();
      sizes.back() = sizes.back() / expandRatio;
      newSizes = rewriter.getI64ArrayAttr(sizes);
    }

    SmallVector<int64_t, 4> dims =
        llvm::to_vector<4>(extractOp.getType().cast<VectorType>().getShape());
    dims.back() = dims.back() / expandRatio;
    VectorType newExtractType =
        VectorType::get(dims, castSrcType.getElementType());

    auto newExtractOp = rewriter.create<vector::ExtractStridedSliceOp>(
        extractOp.getLoc(), newExtractType, castOp.source(), newOffsets,
        newSizes, extractOp.strides());

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

    int64_t castSrcLastDim = castSrcType.getShape().back();
    int64_t castDstLastDim = castDstType.getShape().back();
    // Require casting to less elements for now; other cases to be implemented.
    if (castSrcLastDim < castDstLastDim)
      return failure();

    assert(castSrcLastDim % castDstLastDim == 0);
    int64_t shrinkRatio = castSrcLastDim / castDstLastDim;

    auto insertOp =
        bitcastOp.source().getDefiningOp<vector::InsertStridedSliceOp>();
    if (!insertOp)
      return failure();

    // Only accept all one strides for now.
    if (llvm::any_of(insertOp.strides().getAsValueRange<IntegerAttr>(),
                     [](const APInt &val) { return !val.isOneValue(); }))
      return failure();

    unsigned rank = insertOp.getSourceVectorType().getRank();
    // Require insert op to have the same rank for the source and destination
    // vector; other cases to be implemented.
    if (rank != insertOp.getDestVectorType().getRank())
      return failure();

    ArrayAttr newOffsets = insertOp.offsets();
    assert(newOffsets.size() == rank);
    SmallVector<int64_t, 4> offsets = getIntValueVector(newOffsets);
    if (offsets.back() % shrinkRatio != 0)
      return failure();
    offsets.back() = offsets.back() / shrinkRatio;
    newOffsets = rewriter.getI64ArrayAttr(offsets);

    SmallVector<int64_t, 4> srcDims =
        llvm::to_vector<4>(insertOp.getSourceVectorType().getShape());
    srcDims.back() = srcDims.back() / shrinkRatio;
    VectorType newCastSrcType =
        VectorType::get(srcDims, castDstType.getElementType());

    auto newCastSrcOp = rewriter.create<vector::BitCastOp>(
        bitcastOp.getLoc(), newCastSrcType, insertOp.source());

    SmallVector<int64_t, 4> dstDims =
        llvm::to_vector<4>(insertOp.getDestVectorType().getShape());
    dstDims.back() = dstDims.back() / shrinkRatio;
    VectorType newCastDstType =
        VectorType::get(dstDims, castDstType.getElementType());

    auto newCastDstOp = rewriter.create<vector::BitCastOp>(
        bitcastOp.getLoc(), newCastDstType, insertOp.dest());

    rewriter.replaceOpWithNewOp<vector::InsertStridedSliceOp>(
        bitcastOp, bitcastOp.getType(), newCastSrcOp, newCastDstOp, newOffsets,
        insertOp.strides());

    return success();
  }
};

static Value createCastToIndexLike(PatternRewriter &rewriter, Location loc,
                                   Type targetType, Value value) {
  if (targetType == value.getType())
    return value;

  bool targetIsIndex = targetType.isIndex();
  bool valueIsIndex = value.getType().isIndex();
  if (targetIsIndex ^ valueIsIndex)
    return rewriter.create<arith::IndexCastOp>(loc, targetType, value);

  auto targetIntegerType = targetType.dyn_cast<IntegerType>();
  auto valueIntegerType = value.getType().dyn_cast<IntegerType>();
  assert(targetIntegerType && valueIntegerType &&
         "unexpected cast between types other than integers and index");
  assert(targetIntegerType.getSignedness() == valueIntegerType.getSignedness());

  if (targetIntegerType.getWidth() > valueIntegerType.getWidth())
    return rewriter.create<arith::ExtSIOp>(loc, targetIntegerType, value);
  return rewriter.create<arith::TruncIOp>(loc, targetIntegerType, value);
}

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
                                   bool indexOptimizations, int64_t dim,
                                   Value b, Value *off = nullptr) {
  auto loc = op->getLoc();
  // If we can assume all indices fit in 32-bit, we perform the vector
  // comparison in 32-bit to get a higher degree of SIMD parallelism.
  // Otherwise we perform the vector comparison using 64-bit indices.
  Type idxType =
      indexOptimizations ? rewriter.getI32Type() : rewriter.getI64Type();
  DenseIntElementsAttr indicesAttr;
  if (dim == 0 && indexOptimizations) {
    indicesAttr = DenseIntElementsAttr::get(
        VectorType::get(ArrayRef<int64_t>{}, idxType), ArrayRef<int32_t>{0});
  } else if (dim == 0) {
    indicesAttr = DenseIntElementsAttr::get(
        VectorType::get(ArrayRef<int64_t>{}, idxType), ArrayRef<int64_t>{0});
  } else if (indexOptimizations) {
    indicesAttr = rewriter.getI32VectorAttr(
        llvm::to_vector<4>(llvm::seq<int32_t>(0, dim)));
  } else {
    indicesAttr = rewriter.getI64VectorAttr(
        llvm::to_vector<4>(llvm::seq<int64_t>(0, dim)));
  }
  Value indices = rewriter.create<arith::ConstantOp>(loc, indicesAttr);
  // Add in an offset if requested.
  if (off) {
    Value o = createCastToIndexLike(rewriter, loc, idxType, *off);
    Value ov = rewriter.create<SplatOp>(loc, indices.getType(), o);
    indices = rewriter.create<arith::AddIOp>(loc, ov, indices);
  }
  // Construct the vector comparison.
  Value bound = createCastToIndexLike(rewriter, loc, idxType, b);
  Value bounds = rewriter.create<SplatOp>(loc, indices.getType(), bound);
  return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, indices,
                                        bounds);
}

template <typename ConcreteOp>
struct MaterializeTransferMask : public OpRewritePattern<ConcreteOp> {
public:
  explicit MaterializeTransferMask(MLIRContext *context, bool enableIndexOpt)
      : mlir::OpRewritePattern<ConcreteOp>(context),
        indexOptimizations(enableIndexOpt) {}

  LogicalResult matchAndRewrite(ConcreteOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (!xferOp.hasOutOfBoundsDim())
      return failure();

    if (xferOp.getVectorType().getRank() > 1 ||
        llvm::size(xferOp.indices()) == 0)
      return failure();

    Location loc = xferOp->getLoc();
    VectorType vtp = xferOp.getVectorType();

    // * Create a vector with linear indices [ 0 .. vector_length - 1 ].
    // * Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
    // * Let dim the memref dimension, compute the vector comparison mask
    //   (in-bounds mask):
    //   [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
    //
    // TODO: when the leaf transfer rank is k > 1, we need the last `k`
    //       dimensions here.
    unsigned vecWidth = vtp.getNumElements();
    unsigned lastIndex = llvm::size(xferOp.indices()) - 1;
    Value off = xferOp.indices()[lastIndex];
    Value dim =
        vector::createOrFoldDimOp(rewriter, loc, xferOp.source(), lastIndex);
    Value mask = buildVectorComparison(rewriter, xferOp, indexOptimizations,
                                       vecWidth, dim, &off);

    if (xferOp.mask()) {
      // Intersect the in-bounds with the mask specified as an op parameter.
      mask = rewriter.create<arith::AndIOp>(loc, mask, xferOp.mask());
    }

    rewriter.updateRootInPlace(xferOp, [&]() {
      xferOp.maskMutable().assign(mask);
      xferOp.in_boundsAttr(rewriter.getBoolArrayAttr({true}));
    });

    return success();
  }

private:
  const bool indexOptimizations;
};

/// Conversion pattern for a `vector.create_mask` (0-D and 1-D only).
class VectorCreateMaskOpConversion
    : public OpRewritePattern<vector::CreateMaskOp> {
public:
  explicit VectorCreateMaskOpConversion(MLIRContext *context,
                                        bool enableIndexOpt)
      : mlir::OpRewritePattern<vector::CreateMaskOp>(context),
        indexOptimizations(enableIndexOpt) {}

  LogicalResult matchAndRewrite(vector::CreateMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.getType();
    int64_t rank = dstType.getRank();
    if (rank > 1)
      return failure();
    rewriter.replaceOp(
        op, buildVectorComparison(rewriter, op, indexOptimizations,
                                  rank == 0 ? 0 : dstType.getDimSize(0),
                                  op.getOperand(0)));
    return success();
  }

private:
  const bool indexOptimizations;
};

// Drop inner most contiguous unit dimensions from transfer_read operand.
class DropInnerMostUnitDims : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (readOp.getTransferRank() == 0)
      return failure();

    // TODO: support mask.
    if (readOp.mask())
      return failure();

    auto srcType = readOp.source().getType().dyn_cast<MemRefType>();
    if (!srcType || !srcType.hasStaticShape())
      return failure();

    if (!readOp.permutation_map().isMinorIdentity())
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
    if (srcType.getLayout().getAffineMap().isIdentity()) {
      resultMemrefType = MemRefType::get(
          srcType.getShape().drop_back(dimsToDrop), srcType.getElementType(),
          {}, srcType.getMemorySpaceAsInt());
    } else {
      AffineMap map = srcType.getLayout().getAffineMap();
      int numResultDims = map.getNumDims() - dimsToDrop;
      int numSymbols = map.getNumSymbols();
      for (size_t i = 0; i < dimsToDrop; ++i) {
        int dim = srcType.getRank() - i - 1;
        map = map.replace(rewriter.getAffineDimExpr(dim),
                          rewriter.getAffineConstantExpr(0), numResultDims,
                          numSymbols);
      }
      resultMemrefType = MemRefType::get(
          srcType.getShape().drop_back(dimsToDrop), srcType.getElementType(),
          map, srcType.getMemorySpaceAsInt());
    }

    auto loc = readOp.getLoc();
    SmallVector<int64_t> offsets(srcType.getRank(), 0);
    SmallVector<int64_t> strides(srcType.getRank(), 1);

    ArrayAttr inBoundsAttr =
        readOp.in_bounds()
            ? rewriter.getArrayAttr(
                  readOp.in_boundsAttr().getValue().drop_back(dimsToDrop))
            : ArrayAttr();
    Value rankedReducedView = rewriter.create<memref::SubViewOp>(
        loc, resultMemrefType, readOp.source(), offsets, srcType.getShape(),
        strides);
    auto permMap = getTransferMinorIdentityMap(
        rankedReducedView.getType().cast<ShapedType>(), resultTargetVecType);
    Value result = rewriter.create<vector::TransferReadOp>(
        loc, resultTargetVecType, rankedReducedView,
        readOp.indices().drop_back(dimsToDrop), AffineMapAttr::get(permMap),
        readOp.padding(),
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
  using OpRewritePattern<vector::ScanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ScanOp scanOp,
                                PatternRewriter &rewriter) const override {
    auto loc = scanOp.getLoc();
    VectorType destType = scanOp.getDestType();
    ArrayRef<int64_t> destShape = destType.getShape();
    auto elType = destType.getElementType();
    bool isInt = elType.isIntOrIndex();
    if (!isValidKind(isInt, scanOp.kind()))
      return failure();

    VectorType resType = VectorType::get(destShape, elType);
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resType, rewriter.getZeroAttr(resType));
    int64_t reductionDim = scanOp.reduction_dim();
    bool inclusive = scanOp.inclusive();
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
          loc, reductionType, scanOp.source(), scanOffsets, scanSizes,
          scanStrides);
      Value output;
      if (i == 0) {
        if (inclusive) {
          output = input;
        } else {
          if (initialValueRank == 0) {
            // ShapeCastOp cannot handle 0-D vectors
            output = rewriter.create<vector::BroadcastOp>(
                loc, input.getType(), scanOp.initial_value());
          } else {
            output = rewriter.create<vector::ShapeCastOp>(
                loc, input.getType(), scanOp.initial_value());
          }
        }
      } else {
        Value y = inclusive ? input : lastInput;
        output = genOperator(loc, lastOutput, y, scanOp.kind(), rewriter);
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
    RewritePatternSet &patterns, bool indexOptimizations) {
  patterns.add<VectorCreateMaskOpConversion,
               MaterializeTransferMask<vector::TransferReadOp>,
               MaterializeTransferMask<vector::TransferWriteOp>>(
      patterns.getContext(), indexOptimizations);
}

void mlir::vector::populateShapeCastFoldingPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ShapeCastOpFolder>(patterns.getContext());
}

void mlir::vector::populateBubbleVectorBitCastOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<BubbleDownVectorBitCastForExtract,
               BubbleDownBitCastForStridedSliceExtract,
               BubbleUpBitCastForStridedSliceInsert>(patterns.getContext());
}

void mlir::vector::populateVectorBroadcastLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<BroadcastOpLowering>(patterns.getContext());
}

void mlir::vector::populateVectorMaskOpLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CreateMaskOpLowering, ConstantMaskOpLowering>(
      patterns.getContext());
}

void mlir::vector::populateVectorShapeCastLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ShapeCastOp2DDownCastRewritePattern,
               ShapeCastOp2DUpCastRewritePattern, ShapeCastOpRewritePattern>(
      patterns.getContext());
}

void mlir::vector::populateVectorContractLoweringPatterns(
    RewritePatternSet &patterns, VectorTransformsOptions options) {
  patterns.add<OuterProductOpLowering>(patterns.getContext());
  patterns.add<ContractionOpLowering, ContractionOpToMatmulOpLowering,
               ContractionOpToOuterProductOpLowering>(options,
                                                      patterns.getContext());
}

void mlir::vector::populateVectorTransposeLoweringPatterns(
    RewritePatternSet &patterns, VectorTransformsOptions options) {
  patterns.add<TransposeOpLowering, TransposeOp2DToShuffleLowering>(
      options, patterns.getContext());
}

void mlir::vector::populateVectorReductionToContractPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MultiReduceToContract, CombineContractBroadcast,
               CombineContractTranspose>(patterns.getContext());
}

void mlir::vector::
    populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
        RewritePatternSet &patterns) {
  patterns.add<DropInnerMostUnitDims>(patterns.getContext());
}

void mlir::vector::populateVectorTransferLoweringPatterns(
    RewritePatternSet &patterns, llvm::Optional<unsigned> maxTransferRank) {
  patterns.add<TransferReadToVectorLoadLowering,
               TransferWriteToVectorStoreLowering>(patterns.getContext(),
                                                   maxTransferRank);
  patterns
      .add<VectorLoadToMemrefLoadLowering, VectorStoreToMemrefStoreLowering>(
          patterns.getContext());
}

void mlir::vector::populateVectorScanLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ScanToArithOps>(patterns.getContext());
}
