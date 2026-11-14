//===- SimplifyCFG.cpp - Simplify LLVM control-flow graphs ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/SimplifyCFG.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMSIMPLIFYCFGPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

namespace {
// Running the branch patterns standalone is profitable where full
// canonicalization is not wanted: the generic canonicalizer runs
// every pattern and folds the whole module, which costs more than
// merging blocks and changes operations beyond the branches. Today
// the pass applies only the branch patterns, with folding off, so its
// only effect is block merging, and pipelines that never run
// canonicalization at all can still use the transform through the
// same pattern set.
// Merge forwarding blocks in `func` by applying the llvm.br
// canonicalization patterns.
static LogicalResult mergeBranchBlocks(LLVM::LLVMFuncOp func) {
  RewritePatternSet patterns(func.getContext());
  LLVM::BrOp::getCanonicalizationPatterns(patterns, func.getContext());
  SmallVector<Operation *> ops;
  func.walk([&](LLVM::BrOp op) { ops.push_back(op); });
  // Only the collected branches are rewritten, and folding is off:
  // otherwise the driver would fold operations exposed by a merge and
  // delete dead operand definitions, changing IR beyond block merging.
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);
  config.enableFolding(false);
  return applyOpPatternsGreedily(ops, std::move(patterns), config);
}

// This pass is the home for control-flow simplifications in the LLVM
// dialect. Simplifications should be added here.
struct SimplifyCFGPass
    : public LLVM::impl::LLVMSimplifyCFGPassBase<SimplifyCFGPass> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    if (func.isOptimizeNone())
      return;
    if (failed(mergeBranchBlocks(func)))
      signalPassFailure();
    // Further transforms can be called here.
  }
};
} // namespace
