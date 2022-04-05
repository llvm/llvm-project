//===- VectorInsertExtractStridedSliceRewritePatterns.cpp - Rewrites ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::vector;

// Helper that picks the proper sequence for inserting.
static Value insertOne(PatternRewriter &rewriter, Location loc, Value from,
                       Value into, int64_t offset) {
  auto vectorType = into.getType().cast<VectorType>();
  if (vectorType.getRank() > 1)
    return rewriter.create<InsertOp>(loc, from, into, offset);
  return rewriter.create<vector::InsertElementOp>(
      loc, vectorType, from, into,
      rewriter.create<arith::ConstantIndexOp>(loc, offset));
}

// Helper that picks the proper sequence for extracting.
static Value extractOne(PatternRewriter &rewriter, Location loc, Value vector,
                        int64_t offset) {
  auto vectorType = vector.getType().cast<VectorType>();
  if (vectorType.getRank() > 1)
    return rewriter.create<ExtractOp>(loc, vector, offset);
  return rewriter.create<vector::ExtractElementOp>(
      loc, vectorType.getElementType(), vector,
      rewriter.create<arith::ConstantIndexOp>(loc, offset));
}

/// RewritePattern for InsertStridedSliceOp where source and destination vectors
/// have different ranks.
///
/// When ranks are different, InsertStridedSlice needs to extract a properly
/// ranked vector from the destination vector into which to insert. This pattern
/// only takes care of this extraction part and forwards the rest to
/// [ConvertSameRankInsertStridedSliceIntoShuffle].
///
/// For a k-D source and n-D destination vector (k < n), we emit:
///   1. ExtractOp to extract the (unique) (n-1)-D subvector into which to
///      insert the k-D source.
///   2. k-D -> (n-1)-D InsertStridedSlice op
///   3. InsertOp that is the reverse of 1.
class DecomposeDifferentRankInsertStridedSlice
    : public OpRewritePattern<InsertStridedSliceOp> {
public:
  using OpRewritePattern<InsertStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.getSourceVectorType();
    auto dstType = op.getDestVectorType();

    if (op.getOffsets().getValue().empty())
      return failure();

    auto loc = op.getLoc();
    int64_t rankDiff = dstType.getRank() - srcType.getRank();
    assert(rankDiff >= 0);
    if (rankDiff == 0)
      return failure();

    int64_t rankRest = dstType.getRank() - rankDiff;
    // Extract / insert the subvector of matching rank and InsertStridedSlice
    // on it.
    Value extracted = rewriter.create<ExtractOp>(
        loc, op.getDest(),
        getI64SubArray(op.getOffsets(), /*dropFront=*/0,
                       /*dropBack=*/rankRest));

    // A different pattern will kick in for InsertStridedSlice with matching
    // ranks.
    auto stridedSliceInnerOp = rewriter.create<InsertStridedSliceOp>(
        loc, op.getSource(), extracted,
        getI64SubArray(op.getOffsets(), /*dropFront=*/rankDiff),
        getI64SubArray(op.getStrides(), /*dropFront=*/0));

    rewriter.replaceOpWithNewOp<InsertOp>(
        op, stridedSliceInnerOp.getResult(), op.getDest(),
        getI64SubArray(op.getOffsets(), /*dropFront=*/0,
                       /*dropBack=*/rankRest));
    return success();
  }
};

/// RewritePattern for InsertStridedSliceOp where source and destination vectors
/// have the same rank. For each outermost index in the slice:
///   begin    end             stride
/// [offset : offset+size*stride : stride]
///   1. ExtractOp one (k-1)-D source subvector and one (n-1)-D dest subvector.
///   2. InsertStridedSlice (k-1)-D into (n-1)-D
///   3. the destination subvector is inserted back in the proper place
///   3. InsertOp that is the reverse of 1.
class ConvertSameRankInsertStridedSliceIntoShuffle
    : public OpRewritePattern<InsertStridedSliceOp> {
public:
  using OpRewritePattern<InsertStridedSliceOp>::OpRewritePattern;

  void initialize() {
    // This pattern creates recursive InsertStridedSliceOp, but the recursion is
    // bounded as the rank is strictly decreasing.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(InsertStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.getSourceVectorType();
    auto dstType = op.getDestVectorType();

    if (op.getOffsets().getValue().empty())
      return failure();

    int64_t srcRank = srcType.getRank();
    int64_t dstRank = dstType.getRank();
    assert(dstRank >= srcRank);
    if (dstRank != srcRank)
      return failure();

    if (srcType == dstType) {
      rewriter.replaceOp(op, op.getSource());
      return success();
    }

    int64_t offset =
        op.getOffsets().getValue().front().cast<IntegerAttr>().getInt();
    int64_t size = srcType.getShape().front();
    int64_t stride =
        op.getStrides().getValue().front().cast<IntegerAttr>().getInt();

    auto loc = op.getLoc();
    Value res = op.getDest();

    if (srcRank == 1) {
      int nSrc = srcType.getShape().front();
      int nDest = dstType.getShape().front();
      // 1. Scale source to destType so we can shufflevector them together.
      SmallVector<int64_t> offsets(nDest, 0);
      for (int64_t i = 0; i < nSrc; ++i)
        offsets[i] = i;
      Value scaledSource = rewriter.create<ShuffleOp>(loc, op.getSource(),
                                                      op.getSource(), offsets);

      // 2. Create a mask where we take the value from scaledSource of dest
      // depending on the offset.
      offsets.clear();
      for (int64_t i = 0, e = offset + size * stride; i < nDest; ++i) {
        if (i < offset || i >= e || (i - offset) % stride != 0)
          offsets.push_back(nDest + i);
        else
          offsets.push_back((i - offset) / stride);
      }

      // 3. Replace with a ShuffleOp.
      rewriter.replaceOpWithNewOp<ShuffleOp>(op, scaledSource, op.getDest(),
                                             offsets);

      return success();
    }

    // For each slice of the source vector along the most major dimension.
    for (int64_t off = offset, e = offset + size * stride, idx = 0; off < e;
         off += stride, ++idx) {
      // 1. extract the proper subvector (or element) from source
      Value extractedSource = extractOne(rewriter, loc, op.getSource(), idx);
      if (extractedSource.getType().isa<VectorType>()) {
        // 2. If we have a vector, extract the proper subvector from destination
        // Otherwise we are at the element level and no need to recurse.
        Value extractedDest = extractOne(rewriter, loc, op.getDest(), off);
        // 3. Reduce the problem to lowering a new InsertStridedSlice op with
        // smaller rank.
        extractedSource = rewriter.create<InsertStridedSliceOp>(
            loc, extractedSource, extractedDest,
            getI64SubArray(op.getOffsets(), /* dropFront=*/1),
            getI64SubArray(op.getStrides(), /* dropFront=*/1));
      }
      // 4. Insert the extractedSource into the res vector.
      res = insertOne(rewriter, loc, extractedSource, res, off);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

/// RewritePattern for ExtractStridedSliceOp where source and destination
/// vectors are 1-D. For such cases, we can lower it to a ShuffleOp.
class Convert1DExtractStridedSliceIntoShuffle
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern<ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.getType();

    assert(!op.getOffsets().getValue().empty() && "Unexpected empty offsets");

    int64_t offset =
        op.getOffsets().getValue().front().cast<IntegerAttr>().getInt();
    int64_t size =
        op.getSizes().getValue().front().cast<IntegerAttr>().getInt();
    int64_t stride =
        op.getStrides().getValue().front().cast<IntegerAttr>().getInt();

    auto loc = op.getLoc();
    auto elemType = dstType.getElementType();
    assert(elemType.isSignlessIntOrIndexOrFloat());

    // Single offset can be more efficiently shuffled.
    if (op.getOffsets().getValue().size() != 1)
      return failure();

    SmallVector<int64_t, 4> offsets;
    offsets.reserve(size);
    for (int64_t off = offset, e = offset + size * stride; off < e;
         off += stride)
      offsets.push_back(off);
    rewriter.replaceOpWithNewOp<ShuffleOp>(op, dstType, op.getVector(),
                                           op.getVector(),
                                           rewriter.getI64ArrayAttr(offsets));
    return success();
  }
};

/// RewritePattern for ExtractStridedSliceOp where the source vector is n-D.
/// For such cases, we can rewrite it to ExtractOp/ExtractElementOp + lower
/// rank ExtractStridedSliceOp + InsertOp/InsertElementOp for the n-D case.
class DecomposeNDExtractStridedSlice
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern<ExtractStridedSliceOp>::OpRewritePattern;

  void initialize() {
    // This pattern creates recursive ExtractStridedSliceOp, but the recursion
    // is bounded as the rank is strictly decreasing.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.getType();

    assert(!op.getOffsets().getValue().empty() && "Unexpected empty offsets");

    int64_t offset =
        op.getOffsets().getValue().front().cast<IntegerAttr>().getInt();
    int64_t size =
        op.getSizes().getValue().front().cast<IntegerAttr>().getInt();
    int64_t stride =
        op.getStrides().getValue().front().cast<IntegerAttr>().getInt();

    auto loc = op.getLoc();
    auto elemType = dstType.getElementType();
    assert(elemType.isSignlessIntOrIndexOrFloat());

    // Single offset can be more efficiently shuffled. It's handled in
    // Convert1DExtractStridedSliceIntoShuffle.
    if (op.getOffsets().getValue().size() == 1)
      return failure();

    // Extract/insert on a lower ranked extract strided slice op.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getZeroAttr(elemType));
    Value res = rewriter.create<SplatOp>(loc, dstType, zero);
    for (int64_t off = offset, e = offset + size * stride, idx = 0; off < e;
         off += stride, ++idx) {
      Value one = extractOne(rewriter, loc, op.getVector(), off);
      Value extracted = rewriter.create<ExtractStridedSliceOp>(
          loc, one, getI64SubArray(op.getOffsets(), /* dropFront=*/1),
          getI64SubArray(op.getSizes(), /* dropFront=*/1),
          getI64SubArray(op.getStrides(), /* dropFront=*/1));
      res = insertOne(rewriter, loc, extracted, res, idx);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

void mlir::vector::populateVectorInsertExtractStridedSliceDecompositionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DecomposeDifferentRankInsertStridedSlice,
               DecomposeNDExtractStridedSlice>(patterns.getContext());
}

/// Populate the given list with patterns that convert from Vector to LLVM.
void mlir::vector::populateVectorInsertExtractStridedSliceTransforms(
    RewritePatternSet &patterns) {
  populateVectorInsertExtractStridedSliceDecompositionPatterns(patterns);
  patterns.add<ConvertSameRankInsertStridedSliceIntoShuffle,
               Convert1DExtractStridedSliceIntoShuffle>(patterns.getContext());
}
