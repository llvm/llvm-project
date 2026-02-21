//===- GatherOpPatterns.cpp - Patterns related to tensor.concat lowering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Decompose `tensor.gather` into `linalg.generic`.
///
/// %2 = tensor.gather %0[%1] gather_dims([0]) : (tensor<7x128xf16>,
/// tensor<1x7x1xindex>) -> tensor<1x7x128xf16>
///
/// Becomes
///
/// %empty = tensor.empty() : tensor<1x7x128xf16>
/// %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1,
/// 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types =
/// ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x7x1xindex>)
/// outs(%13 : tensor<1x7x128xf16>) {
///    ^bb0(%in: index, %out: f16):
///      %17 = linalg.index 2 : index
///      %extracted = tensor.extract %0[%in, %17] : tensor<7x128xf16>
///      linalg.yield %extracted : f16
///    } -> tensor<1x7x128xf16>
struct DecomposeTensorGatherOp : public OpRewritePattern<tensor::GatherOp> {
  using OpRewritePattern<tensor::GatherOp>::OpRewritePattern;

  SmallVector<OpFoldResult> getDstMixedSizes(PatternRewriter &rewriter,
                                             Location loc,
                                             tensor::GatherOp gatherOp) const {
    SmallVector<OpFoldResult> dstSize =
        tensor::getMixedSizes(rewriter, loc, gatherOp.getResult());
    SmallVector<OpFoldResult> indexSize =
        tensor::getMixedSizes(rewriter, loc, gatherOp.getIndices());
    SmallVector<OpFoldResult> srcSize =
        tensor::getMixedSizes(rewriter, loc, gatherOp.getSource());
    SmallVector<int64_t> gatherDims(gatherOp.getGatherDims());
    bool isShrinkDst = (indexSize.size() - 1) + srcSize.size() ==
                       dstSize.size() + gatherDims.size();
    for (size_t i = 0; i < indexSize.size() - 1; i++) {
      dstSize[i] = indexSize[i];
    }
    auto cnt = 0;
    for (size_t i = indexSize.size() - 1; i < dstSize.size(); i++) {
      while (isShrinkDst && llvm::find(gatherDims, cnt) != gatherDims.end()) {
        cnt++;
      }
      dstSize[i] = llvm::find(gatherDims, cnt) == gatherDims.end()
                       ? srcSize[cnt]
                       : getAsIndexOpFoldResult(rewriter.getContext(), 1);
      cnt++;
    }
    return dstSize;
  }

  LogicalResult matchAndRewrite(tensor::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(gatherOp);
    Location loc = gatherOp.getLoc();
    SmallVector<int64_t> gatherDims(gatherOp.getGatherDims());

    // create destination tensor for linalg out
    RankedTensorType dstType = gatherOp.getResultType();
    Value dstTensor = rewriter.create<tensor::EmptyOp>(
        loc, getDstMixedSizes(rewriter, loc, gatherOp),
        dstType.getElementType());

    // split index tensor to create the linalg input
    SmallVector<Value> indexTensors;
    Value originIndexTensor = gatherOp.getIndices();
    SmallVector<OpFoldResult> indexTensorSize =
        tensor::getMixedSizes(rewriter, loc, originIndexTensor);
    SmallVector<OpFoldResult> indexTensorStride(
        indexTensorSize.size(),
        getAsIndexOpFoldResult(rewriter.getContext(), 1));
    SmallVector<OpFoldResult> indexTensorOffset(
        indexTensorSize.size(),
        getAsIndexOpFoldResult(rewriter.getContext(), 0));
    indexTensorSize[indexTensorSize.size() - 1] =
        getAsIndexOpFoldResult(rewriter.getContext(), 1);

    for (size_t cnt = 0; cnt < gatherDims.size(); cnt++) {
      indexTensorOffset[indexTensorSize.size() - 1] =
          getAsIndexOpFoldResult(rewriter.getContext(), cnt);
      Value indexTensor = rewriter.create<tensor::ExtractSliceOp>(
          loc, originIndexTensor, indexTensorOffset, indexTensorSize,
          indexTensorStride);
      indexTensors.emplace_back(indexTensor);
    }

    // create the affine map
    SmallVector<AffineMap> affineMaps;
    SmallVector<AffineExpr> dimExprs;
    size_t dstRank = dstType.getShape().size();
    for (unsigned i = 0; i < indexTensorSize.size() - 1; ++i)
      dimExprs.push_back(rewriter.getAffineDimExpr(i));
    dimExprs.push_back(getAffineConstantExpr(0, rewriter.getContext()));

    for (size_t cnt = 0; cnt < gatherDims.size(); cnt++) {
      AffineMap currentMap =
          AffineMap::get(/*dimCount=*/dstRank, /*symbolCount=*/0, dimExprs,
                         rewriter.getContext());
      affineMaps.emplace_back(currentMap);
    }
    affineMaps.emplace_back(rewriter.getMultiDimIdentityMap(dstRank));

    // create iterater types array
    SmallVector<utils::IteratorType> iteratorTypesArray(
        dstRank, utils::IteratorType::parallel);

    // check whether the gather op is valid
    size_t srcRank = gatherOp.getSourceType().getShape().size();
    assert(((indexTensorSize.size() - 1) + srcRank == dstRank ||
            (indexTensorSize.size() - 1) + srcRank ==
                dstRank + gatherDims.size()) &&
           "Expected: index_size - 1 + source_size == dst_size or dst_szie - "
           "gather_size. \n");
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        gatherOp, TypeRange(dstType), indexTensors, ValueRange{dstTensor},
        affineMaps, iteratorTypesArray,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          SmallVector<Value> indexValues(srcRank);
          bool isShrinkDst = (indexTensorSize.size() - 1) + srcRank ==
                             dstRank + gatherDims.size();
          int cnt = 0;
          for (auto i = indexTensorSize.size() - 1; i < dstRank; i++) {
            while (isShrinkDst &&
                   llvm::find(gatherDims, cnt) != gatherDims.end()) {
              cnt++;
            }
            indexValues[cnt] = b.create<linalg::IndexOp>(loc, i);
            cnt++;
          }
          for (auto &&[i, dim] : llvm::enumerate(gatherDims)) {
            indexValues[dim] = args[i];
          }

          Value extract = b.create<tensor::ExtractOp>(loc, gatherOp.getSource(),
                                                      indexValues);
          b.create<linalg::YieldOp>(loc, extract);
        });
    return success();
  }
};

} // namespace

void mlir::tensor::populateDecomposeTensorGatherPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DecomposeTensorGatherOp>(patterns.getContext());
}
