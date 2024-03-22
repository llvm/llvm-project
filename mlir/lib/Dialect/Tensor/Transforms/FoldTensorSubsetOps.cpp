//===- FoldTensorSubsetOps.cpp - Fold tensor subset ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold tensor subset ops with producer / consumers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include <type_traits>

namespace mlir {
namespace tensor {
#define GEN_PASS_DEF_FOLDTENSORSUBSETOPS
#include "mlir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace tensor
} // namespace mlir

using namespace mlir;

static Value getTensorOperand(vector::TransferReadOp op) {
  return op.getSource();
}

static Value getTensorOperand(tensor::InsertSliceOp op) {
  return op.getSource();
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {
/// Merge extract_slice operation with load/transferRead operation.
class TransferReadOfExtractSliceOpFolder final
    : public OpRewritePattern<vector::TransferReadOp> {
public:
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override;
};

/// Merge insert_slice operation with store/transferWriteOp operation.
class InsertSliceOfTransferWriteOpFolder final
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override;
};

/// Merge insert_slice operation with extract_slice operation.
class InsertSliceOfExtractSliceFolder final
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace

template <typename XferOp, typename ExtractOrInsertOp>
static LogicalResult preconditionsFoldExtractOrInsertWithTransferOp(
    RewriterBase &rewriter, XferOp xferOp,
    ExtractOrInsertOp extractOrInsertSliceOp) {
  if (xferOp.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(xferOp, "out of bounds transfer dim");
  if (xferOp.getMask())
    return rewriter.notifyMatchFailure(xferOp, "masked transfer");
  if (!extractOrInsertSliceOp.hasUnitStride()) {
    return rewriter.notifyMatchFailure(
        xferOp, "non-1 stride insert/extract, requires keeping track of "
                "strides, this may result in needing to insert "
                "vector.insert_strided_slice/extract_strided_slice ops");
  }
  return success();
}

LogicalResult TransferReadOfExtractSliceOpFolder::matchAndRewrite(
    vector::TransferReadOp readOp, PatternRewriter &rewriter) const {
  auto extractSliceOp =
      getTensorOperand(readOp).getDefiningOp<tensor::ExtractSliceOp>();
  if (!extractSliceOp)
    return rewriter.notifyMatchFailure(readOp, "not an extract_slice");

  LogicalResult preconditionResult =
      preconditionsFoldExtractOrInsertWithTransferOp(rewriter, readOp,
                                                     extractSliceOp);
  if (failed(preconditionResult))
    return preconditionResult;

  SmallVector<Value> indices(readOp.getIndices().begin(),
                             readOp.getIndices().end());
  SmallVector<Value> sourceIndices;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, readOp.getLoc(), extractSliceOp.getMixedOffsets(),
      extractSliceOp.getMixedStrides(), extractSliceOp.getDroppedDims(),
      indices, sourceIndices);

  rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
      readOp, readOp.getVectorType(), extractSliceOp.getSource(), sourceIndices,
      AffineMapAttr::get(expandDimsToRank(
          readOp.getPermutationMap(), extractSliceOp.getSourceType().getRank(),
          extractSliceOp.getDroppedDims())),
      readOp.getPadding(),
      /*mask=*/Value(), readOp.getInBoundsAttr());

  return success();
}

LogicalResult InsertSliceOfTransferWriteOpFolder::matchAndRewrite(
    tensor::InsertSliceOp insertSliceOp, PatternRewriter &rewriter) const {
  auto writeOp = getTensorOperand(insertSliceOp)
                     .template getDefiningOp<vector::TransferWriteOp>();
  if (!writeOp)
    return rewriter.notifyMatchFailure(insertSliceOp, "not a transfer_write");

  LogicalResult preconditionResult =
      preconditionsFoldExtractOrInsertWithTransferOp(rewriter, writeOp,
                                                     insertSliceOp);
  if (failed(preconditionResult))
    return preconditionResult;

  SmallVector<Value> indices(writeOp.getIndices().begin(),
                             writeOp.getIndices().end());
  SmallVector<Value> sourceIndices;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, writeOp.getLoc(), insertSliceOp.getMixedOffsets(),
      insertSliceOp.getMixedStrides(), insertSliceOp.getDroppedDims(), indices,
      sourceIndices);

  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
      insertSliceOp, writeOp.getValue(), insertSliceOp.getDest(), sourceIndices,
      AffineMapAttr::get(expandDimsToRank(writeOp.getPermutationMap(),
                                          insertSliceOp.getDestType().getRank(),
                                          insertSliceOp.getDroppedDims())),
      writeOp.getInBoundsAttr());

  return success();
}

/// Merge insert_slice operation with extract_slice operation.
///
/// This can be done when the insert_slice op purely expands ranks (adds unit
/// dims) and the extrace_slice drops corresponding unit dims. For example:
///
/// %extracted_slice = tensor.extract_slice %in[0, 0] [1, 8] [1, 1]
///     : tensor<2x8xf32> to tensor<8xf32>
/// %inserted_slice = tensor.insert_slice %extracted_slice
///     into %dest[0, 0] [1, 8] [1, 1]
///     : tensor<8xf32> into tensor<1x8xf32>
///
/// can be folded into:
///
/// %extracted_slice = tensor.extract_slice %in[0, 0] [1, 8] [1, 1]
///     : tensor<2x8xf32> to tensor<1x8xf32>
LogicalResult InsertSliceOfExtractSliceFolder::matchAndRewrite(
    tensor::InsertSliceOp insertSliceOp, PatternRewriter &rewriter) const {
  auto extractSliceOp =
      insertSliceOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!extractSliceOp)
    return failure();

  // Can't fold if the extract_slice op has other users.
  if (!extractSliceOp->hasOneUse())
    return failure();

  // Check if the insert_slice op purely expands ranks (add unit dims).
  if (!isCastLikeInsertSliceOp(insertSliceOp))
    return failure();

  llvm::SmallBitVector extractDroppedDims = extractSliceOp.getDroppedDims();
  llvm::SmallBitVector insertExpandedDims = insertSliceOp.getDroppedDims();
  // Can't fold if the insert_slice op expands to more dims.
  if (extractDroppedDims.size() < insertExpandedDims.size())
    return failure();

  // Try to match the dropped unit dims to the expanded unit dims. This is done
  // by scanning the dims of extract_slice and find the left-most one can match
  // the dim of insert_slice. If a match is found, advance the dim of
  // insert_slice to match the next one.
  unsigned insertDimPos = 0;
  for (unsigned extractDimPos = 0; extractDimPos < extractDroppedDims.size();
       ++extractDimPos) {
    // Matched all expanded dims.
    if (insertDimPos == insertExpandedDims.size())
      break;

    bool isDropped = extractDroppedDims[extractDimPos];
    bool isExpanded = insertExpandedDims[insertDimPos];
    // Match if both sides drop/keep the dim. Advance and match the next dim of
    // insert_slice.
    if (isDropped == isExpanded) {
      insertDimPos += 1;
    } else if (!isDropped && isExpanded) {
      // Not enough dropped unit dims to match the expanded unit dims.
      return failure();
    }
    // If the dim is dropped by extract_slice and not by insert_slice, look the
    // next dim of extract_slice to see if it can match the current dim of
    // insert_slice.
  }
  // Can't match some expanded dims.
  if (insertDimPos != insertExpandedDims.size())
    return failure();

  rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
      insertSliceOp, insertSliceOp.getType(), extractSliceOp.getSource(),
      extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
      extractSliceOp.getMixedStrides());
  rewriter.eraseOp(extractSliceOp);

  return success();
}

template <typename OpTy>
struct InsertSliceOfInsertSliceFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy insertSliceOp,
                                PatternRewriter &rewriter) const override {
    auto sourceInsertSliceOp =
        insertSliceOp.getSource()
            .template getDefiningOp<tensor::InsertSliceOp>();
    if (!sourceInsertSliceOp)
      return failure();

    // TODO: relax unit stride assumption where possible.
    if (!insertSliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "requires unit strides");
    }
    if (!sourceInsertSliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(sourceInsertSliceOp,
                                         "requires unit strides");
    }

    int64_t srcDim = 0;
    llvm::SmallBitVector droppedDims = insertSliceOp.getDroppedDims();
    for (int64_t d = 0, e = insertSliceOp.getDestType().getRank(); d < e; ++d) {
      if (droppedDims[d])
        continue;
      if (insertSliceOp.getMixedSizes()[d] !=
          sourceInsertSliceOp.getMixedSizes()[srcDim++]) {
        return rewriter.notifyMatchFailure(
            sourceInsertSliceOp,
            "requires matching sizes to fold, otherwise a copy is needed");
      }
    }

    // Resolve sizes according to dropped dims.
    SmallVector<OpFoldResult> resolvedSizes;
    // Note: the "insertSlice" case is symmetrical to the extract/subview case:
    // `insertSliceOp` is passed as the "source" and `sourceInsertSliceOp` is
    // passed as the destination to the helper function.
    affine::resolveSizesIntoOpWithSizes(insertSliceOp.getMixedSizes(),
                                        sourceInsertSliceOp.getMixedSizes(),
                                        droppedDims, resolvedSizes);

    // If we are inside an InParallel region, temporarily set the insertion
    // point outside: only tensor.parallel_insert_slice ops are allowed in
    // there.
    if (std::is_same_v<OpTy, tensor::ParallelInsertSliceOp>) {
      rewriter.setInsertionPoint(
          insertSliceOp->template getParentOfType<scf::InParallelOp>());
    }

    // Resolve offsets according to source offsets and strides.
    SmallVector<Value> resolvedOffsets;
    // Note: the "insertSlice" case is symmetrical to the extract/subview case:
    // `insertSliceOp` is passed as the "source" and `sourceInsertSliceOp` is
    // passed as the destination to the helper function.
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, insertSliceOp.getLoc(), insertSliceOp.getMixedOffsets(),
        insertSliceOp.getMixedStrides(), droppedDims,
        sourceInsertSliceOp.getMixedOffsets(), resolvedOffsets);

    // Reset the insertion point.
    rewriter.setInsertionPoint(insertSliceOp);
    // Replace original op.
    rewriter.replaceOpWithNewOp<OpTy>(
        insertSliceOp, sourceInsertSliceOp.getSource(), insertSliceOp.getDest(),
        getAsOpFoldResult(resolvedOffsets), resolvedSizes,
        insertSliceOp.getMixedStrides());

    return success();
  }
};

void tensor::populateFoldTensorSubsetOpPatterns(RewritePatternSet &patterns) {
  populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);
  patterns.add<InsertSliceOfInsertSliceFolder<tensor::InsertSliceOp>,
               InsertSliceOfInsertSliceFolder<tensor::ParallelInsertSliceOp>,
               InsertSliceOfExtractSliceFolder>(patterns.getContext());
}

void tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TransferReadOfExtractSliceOpFolder,
               InsertSliceOfTransferWriteOpFolder>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct FoldTensorSubsetOpsPass final
    : public tensor::impl::FoldTensorSubsetOpsBase<FoldTensorSubsetOpsPass> {
  void runOnOperation() override;
};

} // namespace

void FoldTensorSubsetOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tensor::populateFoldTensorSubsetOpPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> tensor::createFoldTensorSubsetOpsPass() {
  return std::make_unique<FoldTensorSubsetOpsPass>();
}
