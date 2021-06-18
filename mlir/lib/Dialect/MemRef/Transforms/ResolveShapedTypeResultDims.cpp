//===- ResolveShapedTypeResultDims.cpp - Resolve memref.dim ops of result values
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass resolves `memref.dim` operations of result values in terms of
// shapes of their operands using the `InferShapedTypeOpInterface`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

/// Helper method to get the `Value` that is the shape of the `resultIdx`-th
/// result at dimension `dimIndex` from the `ShapedTypeOpInterface`.
/// TODO(ravishankarm): This is better put as a interface utility method
/// somewhere, but that would imply the interface will depend on the `tensor`
/// dialect. Ideally maybe a utility method in the `tensor` dialect.
static Value getResultDimFromShapeInterface(OpBuilder &builder, OpResult result,
                                            int64_t dimIndex) {
  unsigned resultNumber = result.getResultNumber();
  auto shapedTypeOp = dyn_cast<InferShapedTypeOpInterface>(result.getOwner());
  Location loc = result.getOwner()->getLoc();
  if (!shapedTypeOp)
    return nullptr;

  // The interface exposes two methods, one that returns the shape of all the
  // results as `Value` and other that returns the shape as a list of
  // `SmallVector<Value>`. The former takes precedence over the latter. So first
  // check if the op implements the first interface method or the second, and
  // get the value to use appropriately.
  SmallVector<Value> reifiedResultShapes;
  if (succeeded(shapedTypeOp.reifyReturnTypeShapes(
          builder, result.getOwner()->getOperands(), reifiedResultShapes))) {
    if (reifiedResultShapes.size() <= resultNumber)
      return nullptr;
    Value resultShape = reifiedResultShapes[resultNumber];
    auto resultShapeType = resultShape.getType().dyn_cast<RankedTensorType>();
    if (!resultShapeType || !resultShapeType.getElementType().isa<IndexType>())
      return nullptr;
    return builder.create<tensor::ExtractOp>(
        loc, resultShape, builder.createOrFold<ConstantIndexOp>(loc, dimIndex));
  }

  SmallVector<SmallVector<Value>> reifiedResultShapesPerDim;
  if (failed(shapedTypeOp.reifyReturnTypeShapesPerResultDim(
          builder, reifiedResultShapesPerDim)))
    return nullptr;
  if (reifiedResultShapesPerDim.size() <= resultNumber ||
      reifiedResultShapesPerDim[resultNumber].size() !=
          static_cast<size_t>(result.getType().cast<ShapedType>().getRank()))
    return nullptr;
  OpFoldResult valueOrAttr = reifiedResultShapesPerDim[resultNumber][dimIndex];
  if (auto attr = valueOrAttr.dyn_cast<Attribute>())
    return builder.createOrFold<ConstantIndexOp>(
        loc, attr.cast<IntegerAttr>().getInt());
  return valueOrAttr.get<Value>();
}

namespace {
/// Fold dim of an operation that implements the InferShapedTypeOpInterface
struct DimOfShapedTypeOpInterface : public OpRewritePattern<memref::DimOp> {
  using OpRewritePattern<memref::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult dimValue = dimOp.memrefOrTensor().dyn_cast<OpResult>();
    if (!dimValue)
      return failure();
    auto shapedTypeOp =
        dyn_cast<InferShapedTypeOpInterface>(dimValue.getOwner());
    if (!shapedTypeOp)
      return failure();

    Optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();
    Value replacement =
        getResultDimFromShapeInterface(rewriter, dimValue, *dimIndex);
    if (!replacement)
      return failure();
    rewriter.replaceOp(dimOp, replacement);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_CLASSES
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

struct ResolveShapedTypeResultDimsPass final
    : public ResolveShapedTypeResultDimsBase<ResolveShapedTypeResultDimsPass> {
  void runOnOperation() override;
};
} // namespace

void memref::populateResolveShapedTypeResultDimsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DimOfShapedTypeOpInterface>(patterns.getContext());
}

void ResolveShapedTypeResultDimsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                          std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> memref::createResolveShapedTypeResultDimsPass() {
  return std::make_unique<ResolveShapedTypeResultDimsPass>();
}
