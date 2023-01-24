//===- ConvertToDestinationStyle.cpp - Convert non-DPS to DPS ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns to convert non-DPS ops to DPS ops. New
// tensor.empty ops are inserted as a destination. Such tensor.empty can be
// eliminated with "empty tensor elimination", allowing them to bufferize
// without an allocation (assuming there are no further conflicts).
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Lower tensor.generate to linalg.generic.
struct GenerateOpConverter : public OpRewritePattern<GenerateOp> {
  using OpRewritePattern<GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenerateOp generateOp,
                                PatternRewriter &rewriter) const override {
    // Only ops with exactly one block are supported.
    if (!generateOp.getBody().hasOneBlock())
      return failure();

    Location loc = generateOp.getLoc();
    RankedTensorType tensorType = generateOp.getType().cast<RankedTensorType>();

    // Create tensor.empty.
    auto emptyOp = rewriter.create<EmptyOp>(loc, tensorType,
                                            generateOp.getDynamicExtents());

    // Create linalg.generic.
    SmallVector<utils::IteratorType> iteratorTypes(
        tensorType.getRank(), utils::IteratorType::parallel);
    SmallVector<AffineMap> indexingMaps(
        1, rewriter.getMultiDimIdentityMap(tensorType.getRank()));
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, tensorType, /*inputs=*/ValueRange(),
        /*outputs=*/ValueRange{emptyOp.getResult()}, /*indexingMaps=*/
        indexingMaps, iteratorTypes);
    Block *body = rewriter.createBlock(&genericOp->getRegion(0), {},
                                       tensorType.getElementType(), loc);
    rewriter.setInsertionPointToStart(body);
    SmallVector<Value> bbArgReplacements;
    for (int64_t i = 0; i < tensorType.getRank(); ++i)
      bbArgReplacements.push_back(rewriter.create<linalg::IndexOp>(loc, i));
    rewriter.mergeBlocks(&generateOp.getBody().front(), body,
                         bbArgReplacements);

    // Update terminator.
    auto yieldOp = cast<tensor::YieldOp>(body->getTerminator());
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(yieldOp, yieldOp.getValue());

    // Replace tensor.generate.
    rewriter.replaceOp(generateOp, genericOp->getResult(0));
    return success();
  }
};

} // namespace

void linalg::populateConvertToDestinationStylePatterns(
    RewritePatternSet &patterns) {
  patterns.insert<GenerateOpConverter>(patterns.getContext());
}
