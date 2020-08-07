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
  Option<bool> testGenericToVectorPattern{
      *this, "test-contraction-to-vector-patterns",
      llvm::cl::desc("Test a set of patterns that rewrite a linalg contraction "
                     "in vector.contract form"),
      llvm::cl::init(false)};
  Option<bool> testAffineMinSCFCanonicalizationPatterns{
      *this, "test-affine-min-scf-canonicalization-patterns",
      llvm::cl::desc("Test affine-min + scf canonicalization patterns."),
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
      LinalgMarker(Identifier::get("MEM", ctx), Identifier::get("L3", ctx)));
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({200, 300, 400}),
      LinalgMarker(Identifier::get("L3", ctx), Identifier::get("L2", ctx)));
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({20, 30, 40}),
      LinalgMarker(Identifier::get("L2", ctx), Identifier::get("L1", ctx)));
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({2, 3, 4}),
      LinalgMarker(Identifier::get("L1", ctx), Identifier::get("REG", ctx)));

  patterns.insert<LinalgTilingPattern<MatvecOp>>(
      ctx,
      LinalgTilingOptions().setTileSizes({5, 6}).setLoopType(
          LinalgTilingLoopType::ParallelLoops),
      LinalgMarker({}, Identifier::get("L1", ctx)));

  patterns.insert<LinalgTilingPattern<DotOp>>(
      ctx, LinalgTilingOptions().setTileSizes(8000),
      LinalgMarker(ArrayRef<Identifier>{Identifier::get("MEM", ctx),
                                        Identifier::get("L3", ctx),
                                        Identifier::get("L2", ctx)},
                   Identifier::get("REG", ctx)));

  //===--------------------------------------------------------------------===//
  // Linalg tiling and permutation patterns.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx,
      LinalgTilingOptions()
          .setTileSizes({2000, 3000, 4000})
          .setInterchange({1, 2, 0}),
      LinalgMarker(Identifier::get("__with_perm__", ctx),
                   Identifier::get("L2__with_perm__", ctx)));
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx,
      LinalgTilingOptions()
          .setTileSizes({200, 300, 400})
          .setInterchange({1, 0, 2}),
      LinalgMarker(Identifier::get("L2__with_perm__", ctx),
                   Identifier::get("L1__with_perm__", ctx)));
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({20, 30, 40}),
      LinalgMarker(Identifier::get("L1__with_perm__", ctx),
                   Identifier::get("REG__with_perm__", ctx)));

  patterns.insert<LinalgTilingPattern<MatvecOp>>(
      ctx, LinalgTilingOptions().setTileSizes({5, 6}).setInterchange({1, 0}),
      LinalgMarker(Identifier::get("__with_perm__", ctx),
                   Identifier::get("L1__with_perm__", ctx)));

  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx,
      LinalgTilingOptions()
          .setTileSizes({16, 8, 4})
          .setInterchange({1, 2, 0})
          .setLoopType(LinalgTilingLoopType::ParallelLoops),
      LinalgMarker(Identifier::get("par__with_perm__", ctx),
                   Identifier::get("after_par__with_perm__", ctx)));

  //===--------------------------------------------------------------------===//
  // Linalg to loops patterns.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgLoweringPattern<DotOp>>(
      ctx,
      /*loweringType=*/LinalgLoweringType::Loops,
      LinalgMarker(Identifier::get("REG", ctx)));

  //===--------------------------------------------------------------------===//
  // Linalg to vector contraction patterns.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgVectorizationPattern<MatmulOp>,
                  LinalgVectorizationPattern<FillOp>,
                  LinalgVectorizationPattern<CopyOp>,
                  LinalgVectorizationPattern<GenericOp>>(
      ctx, LinalgMarker(Identifier::get("VECTORIZE", ctx)));

  //===--------------------------------------------------------------------===//
  // Linalg generic permutation patterns.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgInterchangePattern<GenericOp>>(
      ctx,
      /*interchangeVector=*/ArrayRef<unsigned>{1, 2, 0},
      LinalgMarker({}, Identifier::get("PERMUTED", ctx)));
  patterns.insert<LinalgInterchangePattern<IndexedGenericOp>>(
      ctx,
      /*interchangeVector=*/ArrayRef<unsigned>{1, 2, 0},
      LinalgMarker({}, Identifier::get("PERMUTED", ctx)));

  //===--------------------------------------------------------------------===//
  // Linalg subview operands promotion.
  //===--------------------------------------------------------------------===//
  patterns.insert<LinalgPromotionPattern<MatmulOp>>(
      ctx, LinalgPromotionOptions().setUseFullTileBuffersByDefault(true),
      LinalgMarker(Identifier::get("_promote_views_", ctx),
                   Identifier::get("_views_promoted_", ctx)));
  patterns.insert<LinalgPromotionPattern<MatmulOp>>(
      ctx,
      LinalgPromotionOptions()
          .setOperandsToPromote({0})
          .setUseFullTileBuffersByDefault(true),
      LinalgMarker(Identifier::get("_promote_first_view_", ctx),
                   Identifier::get("_first_view_promoted_", ctx)));
  patterns.insert<LinalgPromotionPattern<FillOp>>(
      ctx,
      LinalgPromotionOptions()
          .setOperandsToPromote({0})
          .setUseFullTileBuffers({true})
          .setAlignment(32),
      LinalgMarker(Identifier::get("_promote_views_aligned_", ctx),
                   Identifier::get("_views_aligned_promoted_", ctx)));

  applyPatternsAndFoldGreedily(funcOp, patterns);

  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op.removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

static void fillL1TilingAndMatmulToVectorPatterns(
    FuncOp funcOp, StringRef startMarker,
    SmallVectorImpl<OwningRewritePatternList> &patternsVector) {
  MLIRContext *ctx = funcOp.getContext();
  patternsVector.emplace_back(LinalgTilingPattern<MatmulOp>(
      ctx,
      LinalgTilingOptions().setTileSizes({8, 12, 16}).setInterchange({1, 0, 2}),
      LinalgMarker(Identifier::get(startMarker, ctx),
                   Identifier::get("L1", ctx))));

  patternsVector.emplace_back(LinalgPromotionPattern<MatmulOp>(
      ctx, LinalgPromotionOptions().setUseFullTileBuffersByDefault(true),
      LinalgMarker(Identifier::get("L1", ctx), Identifier::get("VEC", ctx))));

  patternsVector.emplace_back(LinalgVectorizationPattern<MatmulOp>(
      ctx, LinalgMarker(Identifier::get("VEC", ctx))));
  patternsVector.back()
      .insert<LinalgVectorizationPattern<FillOp>,
              LinalgVectorizationPattern<CopyOp>>(ctx);
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

static void fillPromotionCallBackPatterns(MLIRContext *ctx,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<LinalgTilingPattern<MatmulOp>>(
      ctx, LinalgTilingOptions().setTileSizes({16, 16, 16}),
      LinalgMarker(Identifier::get("START", ctx),
                   Identifier::get("PROMOTE", ctx)));
  patterns.insert<LinalgPromotionPattern<MatmulOp>>(
      ctx,
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
      LinalgMarker(Identifier::get("PROMOTE", ctx)));
}

static void
applyMatmulToVectorPatterns(FuncOp funcOp,
                            bool testMatmulToVectorPatterns1dTiling,
                            bool testMatmulToVectorPatterns2dTiling) {
  MLIRContext *ctx = funcOp.getContext();
  SmallVector<OwningRewritePatternList, 4> stage1Patterns;
  if (testMatmulToVectorPatterns1dTiling) {
    fillL1TilingAndMatmulToVectorPatterns(funcOp, Identifier::get("START", ctx),
                                          stage1Patterns);
  } else if (testMatmulToVectorPatterns2dTiling) {
    stage1Patterns.emplace_back(LinalgTilingPattern<MatmulOp>(
        ctx,
        LinalgTilingOptions()
            .setTileSizes({768, 264, 768})
            .setInterchange({1, 2, 0}),
        LinalgMarker(Identifier::get("START", ctx),
                     Identifier::get("L2", ctx))));
    fillL1TilingAndMatmulToVectorPatterns(funcOp, Identifier::get("L2", ctx),
                                          stage1Patterns);
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

static void applyContractionToVectorPatterns(FuncOp funcOp) {
  OwningRewritePatternList patterns;
  patterns.insert<LinalgVectorizationPattern<BatchMatmulOp>,
                  LinalgVectorizationPattern<MatmulOp>,
                  LinalgVectorizationPattern<MatvecOp>,
                  LinalgVectorizationPattern<DotOp>,
                  LinalgVectorizationPattern<GenericOp>>(funcOp.getContext());
  applyPatternsAndFoldGreedily(funcOp, patterns);
}

static void applyAffineMinSCFCanonicalizationPatterns(FuncOp funcOp) {
  OwningRewritePatternList foldPattern;
  foldPattern.insert<AffineMinSCFCanonicalizationPattern>(funcOp.getContext());
  // Explicitly walk and apply the pattern locally to avoid more general folding
  // on the rest of the IR.
  funcOp.walk([&foldPattern](AffineMinOp minOp) {
    applyOpPatternsAndFold(minOp, foldPattern);
  });
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
  if (testGenericToVectorPattern)
    return applyContractionToVectorPatterns(getFunction());
  if (testAffineMinSCFCanonicalizationPatterns)
    return applyAffineMinSCFCanonicalizationPatterns(getFunction());
}

namespace mlir {
void registerTestLinalgTransforms() {
  PassRegistration<TestLinalgTransforms> testTransformPatternsPass(
      "test-linalg-transform-patterns",
      "Test Linalg transformation patterns by applying them greedily.");
}
} // namespace mlir
