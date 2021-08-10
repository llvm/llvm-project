//===- CodegenStrategy.cpp - Linalg programmable codegen strategy ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic and helpers to expose Linalg transforms as
// composable rewrite patterns through a programmable CodegenStrategy object.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"

#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-codegen-strategy"

void mlir::linalg::CodegenStrategy::transform(FuncOp func) const {
  MLIRContext *context = func.getContext();
  // Emplace patterns one at a time while also maintaining a simple chained
  // state transition.
  unsigned stepCount = 0;
  SmallVector<FrozenRewritePatternSet, 4> stage1Patterns;
  auto zeroState = Identifier::get(std::to_string(stepCount), context);
  auto currentState = zeroState;
  for (const std::unique_ptr<Transformation> &t : transformationSequence) {
    auto nextState = Identifier::get(std::to_string(++stepCount), context);
    auto marker = (currentState == zeroState)
                      ? linalg::LinalgTransformationFilter(
                            t->filter, ArrayRef<Identifier>{}, nextState)
                      : linalg::LinalgTransformationFilter(
                            t->filter, currentState, nextState);
    stage1Patterns.emplace_back(t->buildRewritePatterns(context, marker));
    currentState = nextState;
  }

  RewritePatternSet stage2Patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  stage2Patterns.add<AffineMinSCFCanonicalizationPattern>(context);

  auto stage3Transforms = [&](Operation *op) {
    // Some of these may be too aggressive as a stage 3 that is applied on each
    // stage 1 application and may have to be split out to post staged patterns
    // application (in which case they could just be passes, TBD).
    if (lateCodegenStrategyOptions.enableLICM) {
      op->walk([&](LoopLikeOpInterface loopLike) {
        LLVM_DEBUG(loopLike.print(llvm::dbgs() << "\nOriginal loop:\n"));
        if (failed(moveLoopInvariantCode(loopLike)))
          llvm_unreachable("unexpected LICM failure");
      });
    }
    promoteSingleIterationLoops(cast<FuncOp>(op));
    if (lateCodegenStrategyOptions.enableHoistRedundantVectorTransfers)
      hoistRedundantVectorTransfers(cast<FuncOp>(op));
    if (lateCodegenStrategyOptions.enableHoistRedundantVectorTransfersOnTensor)
      hoistRedundantVectorTransfersOnTensor(cast<FuncOp>(op));
    return success();
  };
  (void)linalg::applyStagedPatterns(
      func, stage1Patterns, std::move(stage2Patterns), stage3Transforms);

  //===--------------------------------------------------------------------===//
  // Post staged patterns transforms
  //===--------------------------------------------------------------------===//

  // Programmatic splitting of slow/fast path vector transfers.
  if (lateCodegenStrategyOptions.enableVectorTransferPartialRewrite) {
    RewritePatternSet patterns(context);
    patterns.add<vector::VectorTransferFullPartialRewriter>(
        context, vectorTransformsOptions);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }

  // Programmatic controlled lowering of vector.contract only.
  if (lateCodegenStrategyOptions.enableVectorContractLowering) {
    RewritePatternSet vectorContractLoweringPatterns(context);
    vectorContractLoweringPatterns
        .add<ContractionOpToOuterProductOpLowering,
             ContractionOpToMatmulOpLowering, ContractionOpLowering>(
            vectorTransformsOptions, context);
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorContractLoweringPatterns);
    (void)applyPatternsAndFoldGreedily(
        func, std::move(vectorContractLoweringPatterns));
  }

  // Programmatic controlled lowering of vector.transfer only.
  if (lateCodegenStrategyOptions.enableVectorToSCFConversion) {
    RewritePatternSet vectorToLoopsPatterns(context);
    populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                          vectorToSCFOptions);
    (void)applyPatternsAndFoldGreedily(func, std::move(vectorToLoopsPatterns));
  }

  // Ensure we drop the marker in the end.
  func.walk([](LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}
