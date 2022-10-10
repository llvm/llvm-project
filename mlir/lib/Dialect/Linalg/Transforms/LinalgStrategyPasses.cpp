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

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_LINALGSTRATEGYTILEANDFUSEPASS
#define GEN_PASS_DEF_LINALGSTRATEGYTILEPASS
#define GEN_PASS_DEF_LINALGSTRATEGYPADPASS
#define GEN_PASS_DEF_LINALGSTRATEGYDECOMPOSEPASS
#define GEN_PASS_DEF_LINALGSTRATEGYPEELPASS
#define GEN_PASS_DEF_LINALGSTRATEGYLOWERVECTORSPASS
#define GEN_PASS_DEF_LINALGSTRATEGYREMOVEMARKERSPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::vector;
using namespace linalg;

namespace {

/// Configurable pass to apply pattern-based linalg tiling.
struct LinalgStrategyTilePass
    : public impl::LinalgStrategyTilePassBase<LinalgStrategyTilePass> {

  LinalgStrategyTilePass() = default;

  LinalgStrategyTilePass(StringRef opName,
                         mlir::linalg::LinalgTilingOptions opt,
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

  mlir::linalg::LinalgTilingOptions options;
  LinalgTransformationFilter filter;
};

/// Configurable pass to lower vector operations.
struct LinalgStrategyRemoveMarkersPass
    : public impl::LinalgStrategyRemoveMarkersPassBase<
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

/// Create a LinalgStrategyTilePass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyTilePass(StringRef opName,
                                   const LinalgTilingOptions &opt,
                                   const LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyTilePass>(opName, opt, filter);
}

/// Create a LinalgStrategyRemoveMarkersPass.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStrategyRemoveMarkersPass() {
  return std::make_unique<LinalgStrategyRemoveMarkersPass>();
}
