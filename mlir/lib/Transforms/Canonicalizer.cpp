//===- Canonicalizer.cpp - Canonicalize MLIR operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts operations into their canonical forms by
// folding constants, applying operation identity transformations etc.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
/// Canonicalize operations in nested regions.
struct Canonicalizer : public CanonicalizerBase<Canonicalizer> {
  Canonicalizer() = default;
  Canonicalizer(const GreedyRewriteConfig &config,
                ArrayRef<std::string> disabledPatterns,
                ArrayRef<std::string> enabledPatterns) {
    this->topDownProcessingEnabled = config.useTopDownTraversal;
    this->enableRegionSimplification = config.enableRegionSimplification;
    this->maxIterations = config.maxIterations;
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }

  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    patterns = FrozenRewritePatternSet(std::move(owningPatterns),
                                       disabledPatterns, enabledPatterns);
    return success();
  }
  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.useTopDownTraversal = topDownProcessingEnabled;
    config.enableRegionSimplification = enableRegionSimplification;
    config.maxIterations = maxIterations;
    (void)applyPatternsAndFoldGreedily(getOperation(), patterns, config);
  }

  FrozenRewritePatternSet patterns;
};
} // namespace

/// Create a Canonicalizer pass.
std::unique_ptr<Pass> mlir::createCanonicalizerPass() {
  return std::make_unique<Canonicalizer>();
}

/// Creates an instance of the Canonicalizer pass with the specified config.
std::unique_ptr<Pass>
mlir::createCanonicalizerPass(const GreedyRewriteConfig &config,
                              ArrayRef<std::string> disabledPatterns,
                              ArrayRef<std::string> enabledPatterns) {
  return std::make_unique<Canonicalizer>(config, disabledPatterns,
                                         enabledPatterns);
}
