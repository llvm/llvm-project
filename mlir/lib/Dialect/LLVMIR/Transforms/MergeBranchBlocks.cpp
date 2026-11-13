//===- MergeBranchBlocks.cpp - Merge llvm.br forwarding blocks ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/MergeBranchBlocks.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMMERGEBRANCHBLOCKSPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

namespace {
// Running the branch patterns standalone is profitable where full
// canonicalization is not wanted: the generic canonicalizer runs every
// pattern and folds the whole module, which costs more than merging
// blocks and changes operations beyond the branches. This pass applies
// only the branch patterns, with folding off, so its only effect is
// block merging, and pipelines that never run canonicalization at all
// can still use the transform through the same pattern set.
struct MergeBranchBlocksPass
    : public LLVM::impl::LLVMMergeBranchBlocksPassBase<MergeBranchBlocksPass> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    if (func.isOptimizeNone())
      return;
    RewritePatternSet patterns(&getContext());
    LLVM::BrOp::getCanonicalizationPatterns(patterns, &getContext());
    SmallVector<Operation *> ops;
    func.walk([&](LLVM::BrOp op) { ops.push_back(op); });
    // Only the collected branches are rewritten, and folding is off:
    // otherwise the driver would fold operations exposed by a merge and
    // delete dead operand definitions, changing IR beyond block merging.
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    config.enableFolding(false);
    if (failed(applyOpPatternsGreedily(ops, std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace
