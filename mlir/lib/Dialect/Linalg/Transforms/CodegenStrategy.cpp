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
  SmallVector<FrozenRewritePatternList, 4> stage1Patterns;
  auto zeroState = Identifier::get(std::to_string(stepCount), context);
  auto currentState = zeroState;
  for (const std::unique_ptr<Transformation> &t : transformationSequence) {
    auto nextState = Identifier::get(std::to_string(++stepCount), context);
    auto marker = (currentState == zeroState)
                      ? linalg::LinalgMarker({}, nextState)
                      : linalg::LinalgMarker(currentState, nextState);
    stage1Patterns.emplace_back(t->buildRewritePatterns(context, marker));
    currentState = nextState;
  }

  OwningRewritePatternList stage2Patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  stage2Patterns.insert<AffineMinSCFCanonicalizationPattern>(context);

  auto stage3Transforms = [](Operation *op) {
    // Some of these may be too aggressive as a stage 3 that is applied on each
    // stage 1 application and may have to be split out to post staged patterns
    // application (in which case they could just be passes, TBD).
    PassManager pm(op->getContext());
    pm.addPass(createLoopInvariantCodeMotionPass());
    if (failed(pm.run(op->getParentOfType<ModuleOp>())))
      llvm_unreachable("Unexpected failure in cleanup pass pipeline.");
    promoteSingleIterationLoops(cast<FuncOp>(op));
    hoistViewAllocOps(cast<FuncOp>(op));
    hoistRedundantVectorTransfers(cast<FuncOp>(op));
    return success();
  };
  linalg::applyStagedPatterns(func, stage1Patterns, std::move(stage2Patterns),
                              stage3Transforms);

  //===--------------------------------------------------------------------===//
  // Post staged patterns transforms
  //===--------------------------------------------------------------------===//

  ModuleOp module = func->getParentOfType<ModuleOp>();

  // Programmatic splitting of slow/fast path vector transfers.
  OwningRewritePatternList patterns;
  patterns.insert<vector::VectorTransferFullPartialRewriter>(
      context, vectorTransformsOptions);
  applyPatternsAndFoldGreedily(module, std::move(patterns));

  // Programmatic controlled lowering of vector.contract only.
  OwningRewritePatternList vectorContractLoweringPatterns;
  vectorContractLoweringPatterns
      .insert<ContractionOpToOuterProductOpLowering,
              ContractionOpToMatmulOpLowering, ContractionOpLowering>(
          vectorTransformsOptions, context);
  applyPatternsAndFoldGreedily(module,
                               std::move(vectorContractLoweringPatterns));

  // Programmatic controlled lowering of vector.transfer only.
  OwningRewritePatternList vectorToLoopsPatterns;
  populateVectorToSCFConversionPatterns(vectorToLoopsPatterns, context,
                                        vectorToSCFOptions);
  applyPatternsAndFoldGreedily(module, std::move(vectorToLoopsPatterns));

  // Ensure we drop the marker in the end.
  module.walk([](LinalgOp op) {
    op.removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}
