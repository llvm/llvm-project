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

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
#define GEN_PASS_DEF_CANONICALIZERPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// Canonicalize operations in nested regions.
struct Canonicalizer : public impl::CanonicalizerPassBase<Canonicalizer> {
  using impl::CanonicalizerPassBase<Canonicalizer>::CanonicalizerPassBase;
  Canonicalizer(const GreedyRewriteConfig &config,
                ArrayRef<std::string> disabledPatterns,
                ArrayRef<std::string> enabledPatterns)
      : config(config) {
    this->topDownProcessingEnabled = config.getUseTopDownTraversal();
    this->regionSimplifyLevel = config.getRegionSimplificationLevel();
    this->maxIterations = config.getMaxIterations();
    this->maxNumRewrites = config.getMaxNumRewrites();
    this->cseBetweenIterations = config.isCSEBetweenIterationsEnabled();
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // Force-load any dialects named via the `filter-dialects` option. The
    // allocator is resolved later from the MLIRContext's own registry.
    for (const std::string &name : filterDialects)
      registry.addDialectToPreload(StringRef(name));
  }

  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Set the config from possible pass options set in the meantime.
    config.setUseTopDownTraversal(topDownProcessingEnabled);
    config.setRegionSimplificationLevel(regionSimplifyLevel);
    config.setMaxIterations(maxIterations);
    config.setMaxNumRewrites(maxNumRewrites);
    config.enableCSEBetweenIterations(cseBetweenIterations);

    llvm::DenseSet<TypeID> allowedDialects;
    for (const std::string &name : filterDialects) {
      Dialect *dialect = context->getLoadedDialect(name);
      assert(dialect && "filter-dialect should have been preloaded by the "
                        "PassManager via getDependentDialects");
      allowedDialects.insert(dialect->getTypeID());
    }
    auto isAllowed = [&](Dialect *dialect) {
      return allowedDialects.empty() ||
             allowedDialects.contains(dialect->getTypeID());
    };

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      if (isAllowed(dialect))
        dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      if (isAllowed(&op.getDialect()))
        op.getCanonicalizationPatterns(owningPatterns, context);

    patterns = std::make_shared<FrozenRewritePatternSet>(
        std::move(owningPatterns), disabledPatterns, enabledPatterns);
    return success();
  }
  void runOnOperation() override {
    LogicalResult converged =
        applyPatternsGreedily(getOperation(), *patterns, config);
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    if (testConvergence && failed(converged))
      signalPassFailure();
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};
} // namespace

/// Creates an instance of the Canonicalizer pass with the specified config.
std::unique_ptr<Pass>
mlir::createCanonicalizerPass(const GreedyRewriteConfig &config,
                              ArrayRef<std::string> disabledPatterns,
                              ArrayRef<std::string> enabledPatterns) {
  return std::make_unique<Canonicalizer>(config, disabledPatterns,
                                         enabledPatterns);
}
