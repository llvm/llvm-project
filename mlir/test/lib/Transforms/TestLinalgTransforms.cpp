//===- TestLinalgTransforms.cpp - Test Linalg transformation patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Linalg transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestLinalgTransforms
    : public PassWrapper<TestLinalgTransforms, FunctionPass> {
  TestLinalgTransforms() = default;
  TestLinalgTransforms(const TestLinalgTransforms &pass) {}

  void runOnFunction() override;

  Option<bool> testPatterns{*this, "test-patterns",
                            llvm::cl::desc("Test a mixed set of patterns"),
                            llvm::cl::init(false)};
  Option<bool> testMatmulToVectorPatterns1dTiling{
      *this, "test-matmul-to-vector-patterns-tile-1d",
      llvm::cl::desc(
          "Test a fused pass that applies patterns from matmul to vectors via "
          "1-d tiling"),
      llvm::cl::init(false)};
  Option<bool> testMatmulToVectorPatterns2dTiling{
      *this, "test-matmul-to-vector-patterns-tile-2d",
      llvm::cl::desc(
          "Test a fused pass that applies patterns from matmul to vectors via "
          "2-d tiling"),
      llvm::cl::init(false)};
  Option<bool> testPromotionOptions{*this, "test-linalg-promotion-options",
                                    llvm::cl::desc("Test promotion options"),
                                    llvm::cl::init(false)};
  Option<bool> testVectorTransferForwardingPatterns{
      *this, "test-vector-transfer-forwarding-patterns",
      llvm::cl::desc(
          "Test a fused pass that forwards linalg.copy to vector.transfer"),
      llvm::cl::init(false)};
};
} // end anonymous namespace

static void applyPatterns(FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  OwningRewritePatternList patterns;

  //===--------------------------------------------------------------------===//
  // Linalg tiling patterns.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({2000, 3000, 4000}),
      LinalgMarker({"MEM", {}}, "L3"));
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({200, 300, 400}),
      LinalgMarker({"L3"}, "L2"));
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({20, 30, 40}),
      LinalgMarker({"L2"}, "L1"));
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({2, 3, 4}),
      LinalgMarker({"L1"}, "REG"));

  patterns.insert<LinalgTilingPattern<MatvecOp>>(
      ctx,
      LinalgTilingOptions().setTileSizes({5, 6}).setLoopType(
          LinalgTilingLoopType::ParallelLoops),
      LinalgMarker({}, "L1"));

  patterns.insert<LinalgTilingPattern<DotOp>>(
      ctx, LinalgTilingOptions().setTileSizes(8000),
      LinalgMarker({"MEM", "L3", "L2", {}}, "REG"));

  //===--------------------------------------------------------------------===//
  // Linalg tiling and permutation patterns.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx,
      LinalgTilingOptions()
          .setTileSizes({2000, 3000, 4000})
          .setInterchange({1, 2, 0}),
      LinalgMarker({"__with_perm__"}, "L2__with_perm__"));
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx,
      LinalgTilingOptions()
          .setTileSizes({200, 300, 400})
          .setInterchange({1, 0, 2}),
      LinalgMarker({"L2__with_perm__"}, "L1__with_perm__"));
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({20, 30, 40}),
      LinalgMarker({"L1__with_perm__"}, "REG__with_perm__"));

  patterns.insert<LinalgTilingPattern<MatvecOp>>(
      ctx, LinalgTilingOptions().setTileSizes({5, 6}).setInterchange({1, 0}),
      LinalgMarker({"__with_perm__"}, "L1__with_perm__"));

  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx,
      LinalgTilingOptions()
          .setTileSizes({16, 8, 4})
          .setInterchange({1, 2, 0})
          .setLoopType(LinalgTilingLoopType::ParallelLoops),
      LinalgMarker({"par__with_perm__"}, "after_par__with_perm__"));

  //===--------------------------------------------------------------------===//
  // Linalg to loops patterns.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgLoweringPattern<DotOp>>(
      ctx,
      /*loweringType=*/LinalgLoweringType::Loops, LinalgMarker({"REG"}));

  //===--------------------------------------------------------------------===//
  // Linalg to vector contraction patterns.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgVectorizationPattern<MatmulOp>,
                  LinalgVectorizationPattern<FillOp>,
                  LinalgVectorizationPattern<GenericOp>>(
      ctx, LinalgMarker({"VECTORIZE"}));

  //===--------------------------------------------------------------------===//
  // Linalg generic permutation patterns.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgInterchangePattern<GenericOp>>(
      ctx,
      /*interchangeVector=*/ArrayRef<unsigned>{1, 2, 0},
      LinalgMarker({}, "PERMUTED"));
  patterns.insert<LinalgInterchangePattern<IndexedGenericOp>>(
      ctx,
      /*interchangeVector=*/ArrayRef<unsigned>{1, 2, 0},
      LinalgMarker({}, "PERMUTED"));

  //===--------------------------------------------------------------------===//
  // Linalg subview operands promotion.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgPromotionPattern<MatmulOp>>(
      ctx, LinalgPromotionOptions().useFullTileBuffersByDefault(),
      LinalgMarker({"_promote_views_"}, "_views_promoted_"));
  patterns.insert<LinalgPromotionPattern<MatmulOp>>(
      ctx,
      LinalgPromotionOptions()
          .setOperandsToPromote({0})
          .useFullTileBuffersByDefault(),
      LinalgMarker({"_promote_first_view_"}, "_first_view_promoted_"));
  patterns.insert<LinalgPromotionPattern<FillOp>>(
      ctx,
      LinalgPromotionOptions()
          .setOperandsToPromote({0})
          .setUseFullTileBuffers({true})
          .setAlignment(32),
      LinalgMarker({"_promote_views_aligned_"}, "_views_aligned_promoted_"));

  applyPatternsAndFoldGreedily(funcOp, patterns);

  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op.removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

static void fillL1TilingAndMatmulToVectorPatterns(
    FuncOp funcOp, StringRef startMarker,
    SmallVectorImpl<OwningRewritePatternList> &patternsVector) {
  MLIRContext *context = funcOp.getContext();
  patternsVector.emplace_back(LinalgTilingPattern<MatmulOp>(
      context,
      LinalgTilingOptions().setTileSizes({8, 12, 16}).setInterchange({1, 0, 2}),
      LinalgMarker({startMarker}, "L1")));

  patternsVector.emplace_back(LinalgPromotionPattern<MatmulOp>(
      context, LinalgPromotionOptions().useFullTileBuffersByDefault(),
      LinalgMarker({"L1"}, "VEC")));

  patternsVector.emplace_back(
      LinalgVectorizationPattern<MatmulOp>(context, LinalgMarker({"VEC"})));
  patternsVector.back()
      .insert<LinalgVectorizationPattern<FillOp>,
              LinalgVectorizationPattern<CopyOp>>(context);
}

//===----------------------------------------------------------------------===//
// Test promotion callbacks
//===----------------------------------------------------------------------===//

// Allocation call back
static Optional<Value> allocCallBackFn(OpBuilder &b, SubViewOp subView,
                                       ArrayRef<Value> boundingSubViewSize,
                                       OperationFolder *folder) {
  SmallVector<int64_t, 4> shape(boundingSubViewSize.size(), -1);
  return b
      .create<AllocOp>(subView.getLoc(),
                       MemRefType::get(shape,
                                       subView.getType().getElementType(),
                                       /*affineMapComposition =*/{}, 3),
                       boundingSubViewSize)
      .getResult();
}

// Deallocation callback
static LogicalResult deallocCallBackFn(OpBuilder &b, Value buffer) {
  b.create<DeallocOp>(buffer.getLoc(), buffer);
  return success();
}

// Copy in call back
static LogicalResult copyCallBackFn(OpBuilder &b, Value src, Value dst,
                                    bool isOutput) {
  auto floatType = src.getType().cast<MemRefType>().getElementType();
  if (!floatType.isa<FloatType>())
    return failure();
  if (!isOutput)
    b.create<FillOp>(
        src.getLoc(), dst,
        b.create<ConstantOp>(src.getLoc(), FloatAttr::get(floatType, 42.0)));
  b.create<CopyOp>(src.getLoc(), src, dst);
  return success();
}

void fillPromotionCallBackPatterns(MLIRContext *context,
                                   OwningRewritePatternList &patterns) {
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      context, LinalgTilingOptions().setTileSizes({16, 16, 16}),
      LinalgMarker({"START"}, "PROMOTE"));
  patterns.insert<LinalgPromotionPattern<MatmulOp>>(
      context,
      LinalgPromotionOptions()
          .setOperandsToPromote({0, 2})
          .setUseFullTileBuffers({false, false})
          .setAllocationDeallocationFns(allocCallBackFn, deallocCallBackFn)
          .setCopyInOutFns(
              [](OpBuilder &b, Value src, Value dst) -> LogicalResult {
                copyCallBackFn(b, src, dst, false);
                return success();
              },
              [](OpBuilder &b, Value src, Value dst) -> LogicalResult {
                copyCallBackFn(b, src, dst, true);
                return success();
              }),
      LinalgMarker({"PROMOTE"}));
}

static void
applyMatmulToVectorPatterns(FuncOp funcOp,
                            bool testMatmulToVectorPatterns1dTiling,
                            bool testMatmulToVectorPatterns2dTiling) {
  MLIRContext *ctx = funcOp.getContext();
  SmallVector<OwningRewritePatternList, 4> stage1Patterns;
  if (testMatmulToVectorPatterns1dTiling) {
    fillL1TilingAndMatmulToVectorPatterns(funcOp, "START", stage1Patterns);
  } else if (testMatmulToVectorPatterns2dTiling) {
    stage1Patterns.emplace_back(
        LinalgTilingPattern<MatmulOp>(ctx,
                                      LinalgTilingOptions()
                                          .setTileSizes({768, 264, 768})
                                          .setInterchange({1, 2, 0}),
                                      LinalgMarker({"START"}, "L2")));
    fillL1TilingAndMatmulToVectorPatterns(funcOp, "L2", stage1Patterns);
  }
  OwningRewritePatternList stage2Patterns =
      getLinalgTilingCanonicalizationPatterns(ctx);
  applyStagedPatterns(funcOp, stage1Patterns, stage2Patterns);
}

static void applyVectorTransferForwardingPatterns(FuncOp funcOp) {
  OwningRewritePatternList forwardPattern;
  forwardPattern.insert<LinalgCopyVTRForwardingPattern>(funcOp.getContext());
  forwardPattern.insert<LinalgCopyVTWForwardingPattern>(funcOp.getContext());
  applyPatternsAndFoldGreedily(funcOp, forwardPattern);
}

/// Apply transformations specified as patterns.
void TestLinalgTransforms::runOnFunction() {
  auto lambda = [&](void *) {
    getFunction().walk([](LinalgOp op) {
      op.removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
  };
  std::unique_ptr<void, decltype(lambda)> cleanupGuard{(void *)1, lambda};

  if (testPromotionOptions) {
    OwningRewritePatternList patterns;
    fillPromotionCallBackPatterns(&getContext(), patterns);
    applyPatternsAndFoldGreedily(getFunction(), patterns);
    return;
  }
  if (testPatterns)
    return applyPatterns(getFunction());
  if (testMatmulToVectorPatterns1dTiling || testMatmulToVectorPatterns2dTiling)
    return applyMatmulToVectorPatterns(getFunction(),
                                       testMatmulToVectorPatterns1dTiling,
                                       testMatmulToVectorPatterns2dTiling);
  if (testVectorTransferForwardingPatterns)
    return applyVectorTransferForwardingPatterns(getFunction());
}

namespace mlir {
void registerTestLinalgTransforms() {
  PassRegistration<TestLinalgTransforms> testTransformPatternsPass(
      "test-linalg-transform-patterns",
      "Test Linalg transformation patterns by applying them greedily.");
}
} // namespace mlir
