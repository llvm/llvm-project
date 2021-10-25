//===- DynamicPass.cpp - Implementation of a dynamic configurable pass ----===//
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

#include "PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace linalg;

namespace {

/// Configurable pass to apply pattern-based linalg tiling.
struct LinalgStrategyTilePass
    : public LinalgStrategyTilePassBase<LinalgStrategyTilePass> {

  LinalgStrategyTilePass() = default;

  LinalgStrategyTilePass(StringRef opName, LinalgTilingOptions opt,
                         LinalgTransformationFilter filt)
      : options(opt), filter(filt) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnFunction() override {
    auto funcOp = getFunction();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet tilingPattern(funcOp.getContext());
    if (!anchorOpName.empty()) {
      tilingPattern.add<LinalgGenericTilingPattern>(
          anchorOpName, funcOp.getContext(), options, filter);
    } else {
      tilingPattern.add<LinalgGenericTilingPattern>(funcOp.getContext(), filter,
                                                    options);
    }
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(tilingPattern));
  }

  LinalgTilingOptions options;
  LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg generalization.
struct LinalgStrategyGeneralizePass
    : public LinalgStrategyGeneralizePassBase<LinalgStrategyGeneralizePass> {

  LinalgStrategyGeneralizePass() = default;

  LinalgStrategyGeneralizePass(StringRef opName,
                               LinalgTransformationFilter filter)
      : filter(filter) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnFunction() override {
    auto funcOp = getFunction();
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

/// Configurable pass to apply pattern-based linalg generalization.
struct LinalgStrategyInterchangePass
    : public LinalgStrategyInterchangePassBase<LinalgStrategyInterchangePass> {

  LinalgStrategyInterchangePass() = default;

  LinalgStrategyInterchangePass(ArrayRef<int64_t> iteratorInterchange,
                                LinalgTransformationFilter filter)
      : iteratorInterchange(iteratorInterchange.begin(),
                            iteratorInterchange.end()),
        filter(filter) {}

  void runOnFunction() override {
    auto funcOp = getFunction();
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

/// Configurable pass to apply pattern-based linalg promotion.
struct LinalgStrategyPromotePass
    : public LinalgStrategyPromotePassBase<LinalgStrategyPromotePass> {

  LinalgStrategyPromotePass() = default;

  LinalgStrategyPromotePass(StringRef opName, LinalgPromotionOptions opt,
                            LinalgTransformationFilter filt)
      : options(opt), filter(filt) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnFunction() override {
    auto funcOp = getFunction();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet promotionPattern(funcOp.getContext());
    if (!anchorOpName.empty()) {
      promotionPattern.add<LinalgBasePromotionPattern>(
          anchorOpName, funcOp.getContext(), options, filter);
    } else {
      promotionPattern.add<LinalgBasePromotionPattern>(funcOp.getContext(),
                                                       filter, options);
    }
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(promotionPattern));
  }

  LinalgPromotionOptions options;
  LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg vectorization.
struct LinalgStrategyVectorizePass
    : public LinalgStrategyVectorizePassBase<LinalgStrategyVectorizePass> {

  LinalgStrategyVectorizePass() = default;

  LinalgStrategyVectorizePass(StringRef opName, LinalgVectorizationOptions opt,
                              LinalgTransformationFilter filt)
      : options(opt), filter(filt) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnFunction() override {
    auto funcOp = getFunction();
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
    vector::populateVetorReductionToContractPatterns(vectorizationPatterns);
    vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                              linalg::LinalgCopyVTWForwardingPattern>(
        funcOp.getContext(), /*benefit=*/2);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(vectorizationPatterns));
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
      : options(opt), filter(filt) {}

  void runOnFunction() override {
    auto funcOp = getFunction();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();

    if (options.licm) {
      if (funcOp
              ->walk([&](LoopLikeOpInterface loopLike) {
                if (failed(moveLoopInvariantCode(loopLike)))
                  return WalkResult::interrupt();
                return WalkResult::advance();
              })
              .wasInterrupted())
        return signalPassFailure();
    }

    promoteSingleIterationLoops(funcOp);
    if (options.hoistRedundantVectorTransfers)
      hoistRedundantVectorTransfers(funcOp);

    if (options.hoistRedundantVectorTransfersOnTensor)
      hoistRedundantVectorTransfersOnTensor(funcOp);
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
      : options(opt), filter(filt) {}

  void runOnFunction() override {
    auto funcOp = getFunction();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);
    if (options.transferLowering) {
      vector::populateVectorTransferLoweringPatterns(patterns,
                                                     options.maxTransferRank);
    }
    if (options.transferPartialRewrite) {
      patterns.add<vector::VectorTransferFullPartialRewriter>(
          context, options.vectorTransformOptions);
    }
    if (options.contractionLowering) {
      patterns.add<ContractionOpToOuterProductOpLowering,
                   ContractionOpToMatmulOpLowering, ContractionOpLowering>(
          options.vectorTransformOptions, context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    }
    if (options.transferToSCFConversion) {
      populateVectorToSCFConversionPatterns(patterns,
                                            options.vectorTransferToSCFOptions);
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

  void runOnFunction() override {
    auto funcOp = getFunction();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;
    funcOp.walk([](LinalgOp op) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
  }
};
} // namespace

/// Create a LinalgStrategyTilePass.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgStrategyTilePass(StringRef opName, LinalgTilingOptions opt,
                                   LinalgTransformationFilter filter) {
  return std::make_unique<LinalgStrategyTilePass>(opName, opt, filter);
}

/// Create a LinalgStrategyPromotePass.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgStrategyPromotePass(StringRef opName,
                                      LinalgPromotionOptions opt,
                                      LinalgTransformationFilter filter) {
  return std::make_unique<LinalgStrategyPromotePass>(opName, opt, filter);
}

/// Create a LinalgStrategyGeneralizePass.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgStrategyGeneralizePass(StringRef opName,
                                         LinalgTransformationFilter filter) {
  return std::make_unique<LinalgStrategyGeneralizePass>(opName, filter);
}

/// Create a LinalgStrategyInterchangePass.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgStrategyInterchangePass(ArrayRef<int64_t> iteratorInterchange,
                                          LinalgTransformationFilter filter) {
  return std::make_unique<LinalgStrategyInterchangePass>(iteratorInterchange,
                                                         filter);
}

/// Create a LinalgStrategyVectorizePass.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgStrategyVectorizePass(StringRef opName,
                                        LinalgVectorizationOptions opt,
                                        LinalgTransformationFilter filter) {
  return std::make_unique<LinalgStrategyVectorizePass>(opName, opt, filter);
}

/// Create a LinalgStrategyEnablePass.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgStrategyEnablePass(LinalgEnablingOptions opt,
                                     LinalgTransformationFilter filter) {
  return std::make_unique<LinalgStrategyEnablePass>(opt, filter);
}

/// Create a LinalgStrategyLowerVectorsPass.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgStrategyLowerVectorsPass(LinalgVectorLoweringOptions opt,
                                           LinalgTransformationFilter filter) {
  return std::make_unique<LinalgStrategyLowerVectorsPass>(opt, filter);
}

/// Create a LinalgStrategyRemoveMarkersPass.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgStrategyRemoveMarkersPass() {
  return std::make_unique<LinalgStrategyRemoveMarkersPass>();
}
