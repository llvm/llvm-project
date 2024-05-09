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

#include "mlir/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CANONICALIZER
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "canonicalizer"

namespace {
/// Canonicalize operations in nested regions.
struct Canonicalizer : public impl::CanonicalizerBase<Canonicalizer> {
  Canonicalizer() = default;
  Canonicalizer(const GreedyRewriteConfig &config,
                ArrayRef<std::string> disabledPatterns,
                ArrayRef<std::string> enabledPatterns)
      : config(config) {
    this->topDownProcessingEnabled = config.useTopDownTraversal;
    this->enableRegionSimplification = config.enableRegionSimplification;
    this->maxIterations = config.maxIterations;
    this->maxNumRewrites = config.maxNumRewrites;
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }

  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Set the config from possible pass options set in the meantime.
    config.useTopDownTraversal = topDownProcessingEnabled;
    config.enableRegionSimplification = enableRegionSimplification;
    config.maxIterations = maxIterations;
    config.maxNumRewrites = maxNumRewrites;

    LLVM_DEBUG(llvm::dbgs()
               << "[CostModel] Canonicalizer MaxIterations (default):"
               << config.maxIterations << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "[CostModel] Canonicalizer MaxNumRewrites (default):"
               << config.maxNumRewrites << "\n");

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    patterns = std::make_shared<FrozenRewritePatternSet>(
        std::move(owningPatterns), disabledPatterns, enabledPatterns);
    return success();
  }
  void runOnOperation() override {
    Operation *op = getOperation();
    uint32_t cpuID = 0;

    if (isa<ModuleOp>(op)) {
      if (std::optional<int64_t> v =
              DataLayout(llvm::dyn_cast<ModuleOp>(*op))
                  .getCanonicalizerMaxIterations(cpuID)) {
        config.maxIterations = *v;
      }
    } else {
      ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
      if (std::optional<int64_t> v =
              DataLayout(moduleOp).getCanonicalizerMaxIterations(cpuID)) {
        config.maxIterations = *v;
      }
    }

    if (isa<ModuleOp>(op)) {
      if (std::optional<int64_t> v =
              DataLayout(llvm::dyn_cast<ModuleOp>(*op))
                  .getCanonicalizerMaxNumRewrites(cpuID)) {
        config.maxNumRewrites = *v;
      }
    } else {
      ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
      if (std::optional<int64_t> v =
              DataLayout(moduleOp).getCanonicalizerMaxNumRewrites(cpuID)) {
        config.maxNumRewrites = *v;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "[CostModel] Canonicalizer MaxIterations (new):"
                            << config.maxIterations << "\n");
    LLVM_DEBUG(llvm::dbgs() << "[CostModel] Canonicalizer MaxNumRewrites (new):"
                            << config.maxNumRewrites << "\n");

    LogicalResult converged =
        applyPatternsAndFoldGreedily(getOperation(), *patterns, config);
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    if (testConvergence && failed(converged))
      signalPassFailure();
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
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
