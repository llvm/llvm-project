//===- RewriteAsConstant.cpp - Patterns to rewrite tensor ops as constants ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Rewrite tensor.generate with arith.constant if the yielded value is a
/// constant and the tensor type is static.
struct GenerateToConstant : public OpRewritePattern<GenerateOp> {
  using OpRewritePattern<GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenerateOp generateOp,
                                PatternRewriter &rewriter) const override {
    auto tensorType =
        llvm::cast<RankedTensorType>(generateOp.getResult().getType());
    if (!tensorType.hasStaticShape())
      return failure();
    auto terminatorOp =
        cast<tensor::YieldOp>(generateOp.getBody().front().getTerminator());
    Attribute attr;
    if (!matchPattern(terminatorOp.getValue(), m_Constant(&attr)))
      return failure();
    Operation *constantOp =
        rewriter.getContext()
            ->getLoadedDialect<TensorDialect>()
            ->materializeConstant(rewriter,
                                  DenseElementsAttr::get(tensorType, attr),
                                  tensorType, generateOp->getLoc());
    if (!constantOp)
      return failure();
    rewriter.replaceOp(generateOp, constantOp->getResults());
    return success();
  }
};

/// Rewrite tensor.pack with arith.constant if the pack is writing
/// to an empty tensor and the destination shape is static.
struct PackToConstant : OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto constOp = packOp.getSource().getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return failure();
    // Must be a dense constant.
    auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!denseAttr)
      return failure();

    // Bail out if the pack is used as a writing operation i.e.,
    // the destination is not a tensor.empty.
    if (!packOp.getDest().getDefiningOp<tensor::EmptyOp>())
      return rewriter.notifyMatchFailure(packOp,
                                         "expects empty tensor destination");
    // Pack destination must have static shape.
    if (!packOp.getDestType().hasStaticShape())
      return rewriter.notifyMatchFailure(
          packOp, "expects destination with static shape");

    // Pack with padding is not supported currently.
    // TODO: Insert padding values as a part of rewrite.
    if (packOp.getPaddingValue())
      return rewriter.notifyMatchFailure(packOp, "expects no padding value");

    OpBuilder::InsertionGuard guard(rewriter);

    // If it is a splat constant, rewrite the pack directly.
    if (denseAttr.isSplat()) {
      DenseElementsAttr packedDenseShape =
          denseAttr.reshape(packOp.getDestType());
      rewriter.setInsertionPoint(constOp);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(packOp, packedDenseShape);

      return success();
    }

    // Constant contains non-splat dense values.
    // Move the data into a new packed buffer. Each value is placed into its new
    // position as defined by the pack operation.
    ArrayRef<char> srcRawData = denseAttr.getRawData();
    SmallVector<char> destRawData(srcRawData.size());

    int64_t numberOfElements = denseAttr.getNumElements();
    SmallVector<int64_t> strides =
        computeStrides(packOp.getDestType().getShape());

    // Parallelize raw data movement to speedup large constant packing.
    parallelFor(
        packOp.getContext(), 0, numberOfElements,
        [&](size_t destLinearizedIdx) {
          // Step 1: De-linearize destination index.
          // f(lin) = tmp[A][B][C]
          SmallVector<int64_t> destIndices =
              delinearize(destLinearizedIdx, strides);

          // Step 2: Arrange the indexes based on the packing information.
          // Compute inverse of outerDimsPerm to bring the loops into the
          // canonical form tmp[A][B][a][b].
          if (!packOp.getOuterDimsPerm().empty()) {
            SmallVector<int64_t> inversePermutation =
                invertPermutationVector(packOp.getOuterDimsPerm());
            SmallVector<int64_t> tileLoops;
            for (int64_t i = 0; i < packOp.getSourceType().getRank(); i++)
              tileLoops.push_back(destIndices[i]);
            applyPermutationToVector(tileLoops, inversePermutation);

            SmallVector<int64_t> pointLoops;
            for (size_t i = packOp.getSourceType().getRank();
                 i < destIndices.size(); i++) {
              pointLoops.push_back(destIndices[i]);
            }

            destIndices = tileLoops;
            destIndices.append(pointLoops.begin(), pointLoops.end());
          }
          assert(destIndices.size() ==
                 static_cast<size_t>(packOp.getDestType().getRank()));

          // After interchanging the outermost tiled loop we end up in the
          // canonical form tmp[A][B][a][b]. Squash the point loops with the
          // tiled ones.
          llvm::DenseSet<int64_t> tiledLoops(packOp.getInnerDimsPos().begin(),
                                             packOp.getInnerDimsPos().end());
          llvm::DenseMap<int64_t, int64_t> mappingTileToPointLoops;
          // Map the position of the tiled loops with the point one.
          // For example:
          // [A][B] -> [A][B][a][b]
          // entry: [A : 0] [a : 2]
          // entry: [B : 1] [b : 3]
          // [A][B] -> [A][B][b]
          // entry: [B : 1] [b : 2]
          for (auto [idx, tileLoop] : llvm::enumerate(packOp.getInnerDimsPos()))
            mappingTileToPointLoops[tileLoop] = idx;

          SmallVector<int64_t> srcIndices;
          SmallVector<int64_t> tilesSizes = packOp.getStaticTiles();
          int64_t numberOfTileLoops = packOp.getSourceType().getRank();
          size_t tilePosIdx = 0;
          for (int64_t i = 0; i < numberOfTileLoops; i++) {
            if (!tiledLoops.count(i)) {
              // Loop is not tiled.
              srcIndices.push_back(destIndices[i]);
            } else {
              // Loop is tiled, account for the point loop distance.
              srcIndices.push_back(
                  destIndices[i] * tilesSizes[tilePosIdx] +
                  destIndices[numberOfTileLoops + mappingTileToPointLoops[i]]);
              tilePosIdx++;
            }
          }
          assert(srcIndices.size() == static_cast<size_t>(numberOfTileLoops));

          int64_t srcLinearizedIdx = linearize(
              srcIndices, computeStrides(packOp.getSourceType().getShape()));
          assert(srcLinearizedIdx < numberOfElements);

          // Step 3: Do the packing.
          // Copy the source element byte-wise to its packed destination
          // position.
          size_t elementByteSize =
              denseAttr.getRawData().size() / denseAttr.getNumElements();
          for (size_t i = 0; i < elementByteSize; i++) {
            destRawData[destLinearizedIdx * elementByteSize + i] =
                srcRawData[srcLinearizedIdx * elementByteSize + i];
          }
        });

    // Fail gracefully if something went wrong.
    bool detectSpalt = false;
    if (!DenseElementsAttr::isValidRawBuffer(packOp.getDestType(), destRawData,
                                             detectSpalt))
      return rewriter.notifyMatchFailure(
          packOp, "failed to create packed raw data buffer");

    // Replace the pack with a new constant.
    auto packedDenseShape =
        DenseElementsAttr::getFromRawBuffer(packOp.getDestType(), destRawData);
    rewriter.setInsertionPoint(constOp);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(packOp, packedDenseShape);

    return success();
  }
};

} // namespace

void mlir::tensor::populateRewriteAsConstantPatterns(
    RewritePatternSet &patterns) {
  patterns.add<GenerateToConstant, PackToConstant>(patterns.getContext());
}
