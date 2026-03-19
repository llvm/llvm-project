//===- ConcatOpPatterns.cpp - Patterns related to tensor.concat lowering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Decompose `tensor.concat` into `tensor.empty` and a chain of slice inserts.
///
/// %concat = tensor.concat dim(1) %0, %1 :
///         (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
///
/// Becomes
///
/// %empty = tensor.empty() : tensor<2x7xf32>
/// %insert0 = tensor.insert_slice %0 into %empty[0, 0][2, 3][1, 1]
/// %concat = tensor.insert_slice %1 into %insert0[0, 3][2, 4][1, 1]
struct DecomposeTensorConcatOp : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern<ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Value>> decomposed =
        concatOp.decomposeOperation(rewriter);
    if (failed(decomposed)) {
      return rewriter.notifyMatchFailure(
          concatOp, "failed to get the decomposed insert slices");
    }
    rewriter.replaceOp(concatOp, decomposed.value()[0]);
    return success();
  }
};

/// Forward the destination tensor of concat generated tensor.insert_slice ops
/// into single-use destination-style tensor producers. This avoids creating a
/// producer on a temporary tensor that is immediately copied into the concat
/// result tensor.
///
/// Before:
/// %small = tensor.empty() : tensor<4xf32>
/// %fill = linalg.fill ins(%cst : f32) outs(%small : tensor<4xf32>)
///     -> tensor<4xf32>
/// %init = tensor.empty() : tensor<8xf32>
/// %insert0 = tensor.insert_slice %fill into %init[0] [4] [1]
///     : tensor<4xf32> into tensor<8xf32>
/// %insert1 = tensor.insert_slice %arg0 into %insert0[4] [4] [1]
///     : tensor<4xf32> into tensor<8xf32>
///
/// After:
/// %init = tensor.empty() : tensor<8xf32>
/// %slice = tensor.extract_slice %init[0] [4] [1]
///     : tensor<8xf32> to tensor<4xf32>
/// %fill = linalg.fill ins(%cst : f32) outs(%slice : tensor<4xf32>)
///     -> tensor<4xf32>
/// %insert0 = tensor.insert_slice %fill into %init[0] [4] [1]
///     : tensor<4xf32> into tensor<8xf32>
/// %insert1 = tensor.insert_slice %arg0 into %insert0[4] [4] [1]
///     : tensor<4xf32> into tensor<8xf32>
struct ForwardConcatInsertSliceDest : public OpRewritePattern<InsertSliceOp> {
  using OpRewritePattern<InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    // Only rewrite when the insert source is an SSA result with a single use.
    Value source = insertOp.getSource();
    auto sourceResult = dyn_cast<OpResult>(source);
    if (!sourceResult || !source.hasOneUse())
      return failure();

    // Restrict to concat-style insert chains where the destination is either
    // the initial tensor.empty or a previous tensor.insert_slice result.
    Operation *destDef = insertOp.getDest().getDefiningOp();
    if (!isa_and_present<EmptyOp, InsertSliceOp>(destDef))
      return failure();

    // The source producer must be destination-style on tensors so we can
    // retarget its tied output to a slice of the final concat destination.
    auto producer = source.getDefiningOp<DestinationStyleOpInterface>();
    if (!producer || !producer.hasPureTensorSemantics())
      return failure();

    if (producer->getNumResults() != 1)
      return failure();

    OpOperand *tiedInit = producer.getTiedOpOperand(sourceResult);
    if (!tiedInit)
      return failure();

    auto sourceType = dyn_cast<RankedTensorType>(source.getType());
    if (!sourceType || !isa<RankedTensorType>(insertOp.getDest().getType()))
      return failure();

    auto mixedOffsets = insertOp.getMixedOffsets();
    auto mixedSizes = insertOp.getMixedSizes();
    auto mixedStrides = insertOp.getMixedStrides();

    auto extractedInit = tiedInit->get().getDefiningOp<ExtractSliceOp>();
    if (extractedInit && extractedInit.getSource() == insertOp.getDest() &&
        llvm::equal(extractedInit.getMixedOffsets(), mixedOffsets) &&
        llvm::equal(extractedInit.getMixedSizes(), mixedSizes) &&
        llvm::equal(extractedInit.getMixedStrides(), mixedStrides)) {
      return failure();
    }

    // Extract slice from the final destination
    Value extractedDest = ExtractSliceOp::create(
        rewriter, insertOp.getLoc(), sourceType, insertOp.getDest(),
        mixedOffsets, mixedSizes, mixedStrides);

    IRMapping mapping;
    mapping.map(tiedInit->get(), extractedDest);
    Operation *newProducer = rewriter.clone(*producer, mapping);
    Value newSource = newProducer->getResult(sourceResult.getResultNumber());

    // Rebuild insert_slice with the retargeted producer result, then erase the
    // original producer (guaranteed to have a single use)
    Value newInsert = InsertSliceOp::create(
        rewriter, insertOp.getLoc(), newSource, insertOp.getDest(),
        mixedOffsets, mixedSizes, mixedStrides);
    rewriter.replaceOp(insertOp, newInsert);
    rewriter.eraseOp(producer.getOperation());
    return success();
  }
};

} // namespace

void mlir::tensor::populateDecomposeTensorConcatPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DecomposeTensorConcatOp>(patterns.getContext());
}

void mlir::tensor::populateForwardConcatInsertSliceDestPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ForwardConcatInsertSliceDest>(patterns.getContext());
}
