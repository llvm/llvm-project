//===- LinalgStrategyPasses.cpp - Implementation of Linalg passes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a configurable pass that can apply patterns liberally
// and be plugged in a pass pipeline.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::vector;
using namespace linalg;

namespace {

/// Configurable pass to apply pattern-based tiling and fusion.
struct LinalgStrategyTileAndFusePass
    : public LinalgStrategyTileAndFusePassBase<LinalgStrategyTileAndFusePass> {

  LinalgStrategyTileAndFusePass() = default;

  LinalgStrategyTileAndFusePass(StringRef opName,
                                LinalgTilingAndFusionOptions opt,
                                LinalgTransformationFilter filt)
      : options(std::move(opt)), filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet tilingAndFusionPattern(funcOp.getContext());
    if (!anchorOpName.empty()) {
      tilingAndFusionPattern.add<LinalgTileAndFuseTensorOpsPattern>(
          anchorOpName, funcOp.getContext(), options, filter);
    } else {
      tilingAndFusionPattern.add<LinalgTileAndFuseTensorOpsPattern>(
          funcOp.getContext(), options, filter);
    }
    // Search the root operation using bottom up traversal.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = false;
    (void)applyPatternsAndFoldGreedily(
        funcOp, std::move(tilingAndFusionPattern), config);
  }

  LinalgTilingAndFusionOptions options;
  LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg tiling.
struct LinalgStrategyTilePass
    : public LinalgStrategyTilePassBase<LinalgStrategyTilePass> {

  LinalgStrategyTilePass() = default;

  LinalgStrategyTilePass(StringRef opName, LinalgTilingOptions opt,
                         LinalgTransformationFilter filt)
      : options(std::move(opt)), filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet tilingPattern(ctx);
    if (!anchorOpName.empty())
      tilingPattern.add<LinalgTilingPattern>(anchorOpName, ctx, options,
                                             filter);
    else
      tilingPattern.add<LinalgTilingPattern>(ctx, options, filter);
    if (anchorOpName == tensor::PadOp::getOperationName())
      populatePadTensorTilingPatterns(tilingPattern, options);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(tilingPattern));
  }

  LinalgTilingOptions options;
  LinalgTransformationFilter filter;
};

/// Configurable pass to apply hoisting and padding.
struct LinalgStrategyPadPass
    : public LinalgStrategyPadPassBase<LinalgStrategyPadPass> {

  LinalgStrategyPadPass() = default;

  LinalgStrategyPadPass(StringRef opName, LinalgPaddingOptions opt,
                        LinalgTransformationFilter filt)
      : options(std::move(opt)), filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet paddingPattern(funcOp.getContext());
    if (!anchorOpName.empty()) {
      paddingPattern.add<LinalgPaddingPattern>(
          anchorOpName, funcOp.getContext(), options, filter);
    } else {
      paddingPattern.add<LinalgPaddingPattern>(funcOp.getContext(), options,
                                               filter);
    }
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(paddingPattern));
  }

  LinalgPaddingOptions options;
  LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg generalization.
struct LinalgStrategyGeneralizePass
    : public LinalgStrategyGeneralizePassBase<LinalgStrategyGeneralizePass> {

  LinalgStrategyGeneralizePass() = default;

  LinalgStrategyGeneralizePass(StringRef opName,
                               LinalgTransformationFilter filter)
      : filter(std::move(filter)) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet generalizationPattern(funcOp.getContext());
    if (!anchorOpName.empty()) {
      generalizationPattern.add<LinalgGeneralizationPattern>(
          anchorOpName, funcOp.getContext(), filter);
    } else {
      generalizationPattern.add<LinalgGeneralizationPattern>(
          funcOp.getContext(), filter);
    }
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(generalizationPattern))))
      signalPassFailure();
  }

  LinalgTransformationFilter filter;
};

/// Configurable pass to apply lowering of coarser-grained named linalg ops into
/// finer-grained named versions.
struct LinalgStrategyDecomposePass
    : public LinalgStrategyDecomposePassBase<LinalgStrategyDecomposePass> {

  LinalgStrategyDecomposePass() = default;

  LinalgStrategyDecomposePass(LinalgTransformationFilter filter)
      : filter(std::move(filter)) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;
    RewritePatternSet decompositionPattern(funcOp.getContext());
    populateDecomposeConvolutionPatterns(decompositionPattern, filter);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(decompositionPattern))))
      signalPassFailure();
  }

  LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg generalization.
struct LinalgStrategyInterchangePass
    : public LinalgStrategyInterchangePassBase<LinalgStrategyInterchangePass> {

  LinalgStrategyInterchangePass() = default;

  LinalgStrategyInterchangePass(ArrayRef<int64_t> iteratorInterchange,
                                LinalgTransformationFilter filter)
      : iteratorInterchange(iteratorInterchange.begin(),
                            iteratorInterchange.end()),
        filter(std::move(filter)) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    SmallVector<unsigned> interchangeVector(iteratorInterchange.begin(),
                                            iteratorInterchange.end());
    RewritePatternSet interchangePattern(funcOp.getContext());
    interchangePattern.add<GenericOpInterchangePattern>(
        funcOp.getContext(), interchangeVector, filter);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(interchangePattern))))
      signalPassFailure();
  }

  SmallVector<int64_t> iteratorInterchange;
  LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg peeling.
struct LinalgStrategyPeelPass
    : public LinalgStrategyPeelPassBase<LinalgStrategyPeelPass> {

  LinalgStrategyPeelPass() = default;

  LinalgStrategyPeelPass(StringRef opName, LinalgPeelOptions opt,
                         LinalgTransformationFilter filt)
      : options(std::move(opt)), filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet peelingPatterns(funcOp.getContext());
    if (!anchorOpName.empty()) {
      peelingPatterns.add<LinalgPeelingPattern>(
          anchorOpName, funcOp.getContext(), options, filter);
    } else {
      peelingPatterns.add<LinalgPeelingPattern>(funcOp.getContext(), filter,
                                                options);
    }
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(peelingPatterns))))
      return signalPassFailure();
  }

  LinalgPeelOptions options;
  LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg vectorization.
struct LinalgStrategyVectorizePass
    : public LinalgStrategyVectorizePassBase<LinalgStrategyVectorizePass> {

  LinalgStrategyVectorizePass() = default;

  LinalgStrategyVectorizePass(StringRef opName, LinalgVectorizationOptions opt,
                              LinalgTransformationFilter filt,
                              bool padVectorize = false)
      : options(opt), filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
    this->vectorizePadding.setValue(padVectorize);
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet vectorizationPatterns(funcOp.getContext());
    if (!anchorOpName.empty()) {
      vectorizationPatterns.add<LinalgVectorizationPattern>(
          anchorOpName, funcOp.getContext(), options, filter);
    } else {
      vectorizationPatterns.add<LinalgVectorizationPattern>(funcOp.getContext(),
                                                            filter, options);
    }
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizationPatterns);
    vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
    vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                              linalg::LinalgCopyVTWForwardingPattern>(
        funcOp.getContext(), /*benefit=*/2);
    TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                funcOp.getContext());
    TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                 funcOp.getContext());
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(vectorizationPatterns));

    // Apply the pad tensor op vectorization separately to avoid running the
    // GenericPadOpVectorizationPattern too early.
    // TODO: Improve once we have better infrastructure to control pattern
    // application.
    if (vectorizePadding) {
      RewritePatternSet patterns(funcOp.getContext());
      linalg::populatePadOpVectorizationPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }
  }

  LinalgVectorizationOptions options;
  LinalgTransformationFilter filter;
};

/// Configurable pass to enable the application of other pattern-based linalg
/// passes.
struct LinalgStrategyEnablePass
    : public LinalgStrategyEnablePassBase<LinalgStrategyEnablePass> {

  LinalgStrategyEnablePass(LinalgEnablingOptions opt,
                           LinalgTransformationFilter filt)
      : options(opt), filter(std::move(filt)) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();

    if (options.licm) {
      funcOp->walk([&](LoopLikeOpInterface loopLike) {
        moveLoopInvariantCode(loopLike);
      });
    }

    // Gathers all innermost loops through a post order pruned walk.
    funcOp.walk([](Operation *op) {
      if (auto forOp = dyn_cast<AffineForOp>(op))
        (void)promoteIfSingleIteration(forOp);
      else if (auto forOp = dyn_cast<scf::ForOp>(op))
        (void)promoteIfSingleIteration(forOp);
    });
    if (options.hoistRedundantVectorTransfers)
      hoistRedundantVectorTransfers(funcOp);

    if (options.hoistRedundantVectorTransfersOnTensor)
      hoistRedundantVectorTransfersOnTensor(funcOp);

    // Run CSE to cleanup after canonicalization.
    OpPassManager dynamicPM("func.func");
    dynamicPM.addPass(createCSEPass());
    if (failed(runPipeline(dynamicPM, funcOp)))
      return signalPassFailure();
  }

  LinalgEnablingOptions options;
  LinalgTransformationFilter filter;
};

/// Configurable pass to lower vector operations.
struct LinalgStrategyLowerVectorsPass
    : public LinalgStrategyLowerVectorsPassBase<
          LinalgStrategyLowerVectorsPass> {

  LinalgStrategyLowerVectorsPass(LinalgVectorLoweringOptions opt,
                                 LinalgTransformationFilter filt)
      : options(opt), filter(std::move(filt)) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    // In a progressive lowering of vectors, this would be the 1st step.
    if (options.contractionLowering) {
      patterns.add<ContractionOpToOuterProductOpLowering,
                   ContractionOpToMatmulOpLowering, ContractionOpLowering>(
          options.vectorTransformOptions, context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    }
    // In a progressive lowering of vectors, this would be the 2nd step.
    if (options.multiReductionLowering) {
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns,
          options.vectorTransformOptions.vectorMultiReductionLowering);
    }
    // In a progressive lowering of vectors, this would be the 3rd step.
    if (options.transferPartialRewrite) {
      patterns.add<vector::VectorTransferFullPartialRewriter>(
          context, options.vectorTransformOptions);
    }
    // In a progressive lowering of vectors, this would be the 4th step.
    if (options.transferLowering) {
      vector::populateVectorTransferLoweringPatterns(patterns,
                                                     options.maxTransferRank);
    }
    // In a progressive lowering of vectors, this would be the 5th step.
    if (options.transferToSCFConversion) {
      populateVectorToSCFConversionPatterns(
          patterns, options.vectorTransferToSCFOptions.setTargetRank(
                        options.maxTransferRank));
    }
    // In a progressive lowering of vectors, this would be the 6th step.
    if (options.shapeCastLowering) {
      vector::populateVectorShapeCastLoweringPatterns(patterns);
    }
    // In a progressive lowering of vectors, this would be the 7th step.
    if (options.transposeLowering) {
      vector::populateVectorTransposeLoweringPatterns(
          patterns, options.vectorTransformOptions);
      if (options.avx2Lowering)
        x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
            patterns, options.avx2LoweringOptions, /*benefit=*/10);
    }
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  LinalgVectorLoweringOptions options;
  LinalgTransformationFilter filter;
};

/// Configurable pass to lower vector operations.
struct LinalgStrategyRemoveMarkersPass
    : public LinalgStrategyRemoveMarkersPassBase<
          LinalgStrategyRemoveMarkersPass> {

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;
    funcOp.walk([](LinalgOp op) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
  }
};
} // namespace

/// Create a LinalgStrategyTileAndFusePass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyTileAndFusePass(
    StringRef opName, const LinalgTilingAndFusionOptions &options,
    const LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyTileAndFusePass>(opName, options,
                                                         filter);
}

/// Create a LinalgStrategyTilePass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyTilePass(StringRef opName,
                                   const LinalgTilingOptions &opt,
                                   const LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyTilePass>(opName, opt, filter);
}

/// Create a LinalgStrategyPadPass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyPadPass(StringRef opName,
                                  const LinalgPaddingOptions &opt,
                                  const LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyPadPass>(opName, opt, filter);
}

/// Create a LinalgStrategyGeneralizePass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyGeneralizePass(
    StringRef opName, const LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyGeneralizePass>(opName, filter);
}

/// Create a LinalgStrategyDecomposePass.
// TODO: if/when we need finer control add an `opName` parameter.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyDecomposePass(
    const LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyDecomposePass>(filter);
}

/// Create a LinalgStrategyInterchangePass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyInterchangePass(
    ArrayRef<int64_t> iteratorInterchange,
    const LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyInterchangePass>(iteratorInterchange,
                                                         filter);
}

/// Create a LinalgStrategyPeelPass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyPeelPass(StringRef opName,
                                   const LinalgPeelOptions &opt,
                                   const LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyPeelPass>(opName, opt, filter);
}

/// Create a LinalgStrategyVectorizePass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyVectorizePass(
    StringRef opName, LinalgVectorizationOptions opt,
    const LinalgTransformationFilter &filter, bool padVectorize) {
  return std::make_unique<LinalgStrategyVectorizePass>(opName, opt, filter,
                                                       padVectorize);
}

/// Create a LinalgStrategyEnablePass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyEnablePass(LinalgEnablingOptions opt,
                                     const LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyEnablePass>(opt, filter);
}

/// Create a LinalgStrategyLowerVectorsPass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyLowerVectorsPass(
    LinalgVectorLoweringOptions opt, const LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyLowerVectorsPass>(opt, filter);
}

/// Create a LinalgStrategyRemoveMarkersPass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyRemoveMarkersPass() {
  return std::make_unique<LinalgStrategyRemoveMarkersPass>();
}
