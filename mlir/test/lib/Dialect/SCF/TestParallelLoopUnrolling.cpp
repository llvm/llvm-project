//=== TestParallelLoopUnrolling.cpp - loop unrolling test pass ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to unroll loops by a specified unroll factor.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

static unsigned getNestingDepth(Operation *op) {
  Operation *currOp = op;
  unsigned depth = 0;
  while ((currOp = currOp->getParentOp())) {
    if (isa<scf::ParallelOp>(currOp))
      depth++;
  }
  return depth;
}

struct TestParallelLoopUnrollingPass
    : public PassWrapper<TestParallelLoopUnrollingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestParallelLoopUnrollingPass)

  StringRef getArgument() const final { return "test-parallel-loop-unrolling"; }
  StringRef getDescription() const final {
    return "Tests parallel loop unrolling transformation";
  }
  TestParallelLoopUnrollingPass() = default;
  TestParallelLoopUnrollingPass(const TestParallelLoopUnrollingPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    if (unrollFactors.empty()) {
      emitError(UnknownLoc::get(&getContext()),
                "missing `unroll-factors` pass option");
      signalPassFailure();
      return;
    }
    if (llvm::is_contained(unrollFactors, 0)) {
      emitError(UnknownLoc::get(&getContext()),
                "unroll factors must be non-zero");
      signalPassFailure();
      return;
    }

    SmallVector<scf::ParallelOp, 4> loops;
    getOperation()->walk([&](scf::ParallelOp parLoop) {
      if (getNestingDepth(parLoop) == loopDepth)
        loops.push_back(parLoop);
    });
    auto annotateFn = [this](unsigned i, Operation *op, OpBuilder b) {
      if (annotateLoop) {
        op->setDiscardableAttr("unrolled_iteration", b.getUI32IntegerAttr(i));
      }
    };
    // Combines the values of the unrolled iterations with a cast.
    auto combineWithCast = [](RewriterBase &rewriter, scf::ReduceOp reduceOp,
                              ArrayRef<SmallVector<Value>> iterReduced) {
      SmallVector<Value> reduced;
      for (unsigned idx = 0, e = iterReduced.front().size(); idx < e; ++idx) {
        SmallVector<Value> inputs;
        for (ArrayRef<Value> iter : iterReduced)
          inputs.push_back(iter[idx]);
        reduced.push_back(
            UnrealizedConversionCastOp::create(rewriter, reduceOp.getLoc(),
                                               inputs[0].getType(), inputs)
                .getResult(0));
      }
      return reduced;
    };
    function_ref<SmallVector<Value>(RewriterBase &, scf::ReduceOp,
                                    ArrayRef<SmallVector<Value>>)>
        combineFn = nullptr;
    if (combineCast)
      combineFn = combineWithCast;
    PatternRewriter rewriter(getOperation()->getContext());
    for (auto loop : loops) {
      SmallPtrSet<Operation *, 8> srcOps;
      for (Operation &op : loop.getBody()->without_terminator())
        srcOps.insert(&op);
      IRMapping clonedToSrcOpsMap;
      if (failed(parallelLoopUnrollByFactors(loop, unrollFactors, rewriter,
                                             annotateFn, &clonedToSrcOpsMap,
                                             combineFn)))
        continue;
      for (auto [cloned, src] : clonedToSrcOpsMap.getOperationMap())
        if (mapClonedOps && srcOps.contains(src))
          cloned->setDiscardableAttr("cloned_from_original",
                                     rewriter.getUnitAttr());
    }
  }

  ListOption<uint64_t> unrollFactors{
      *this, "unroll-factors",
      llvm::cl::desc(
          "Unroll factors for each parallel loop dim. If fewer factors than "
          "loop dims are provided, they are applied to the inner dims.")};
  Option<unsigned> loopDepth{*this, "loop-depth", llvm::cl::desc("Loop depth."),
                             llvm::cl::init(0)};
  Option<bool> annotateLoop{*this, "annotate",
                            llvm::cl::desc("Annotate unrolled iterations."),
                            llvm::cl::init(false)};
  Option<bool> mapClonedOps{
      *this, "map-cloned-ops",
      llvm::cl::desc("Mark the clones mapped to an original op."),
      llvm::cl::init(false)};
  Option<bool> combineCast{
      *this, "combine-with-cast",
      llvm::cl::desc("Combine the unrolled reductions with a cast."),
      llvm::cl::init(false)};
};
} // namespace

namespace mlir {
namespace test {
void registerTestParallelLoopUnrollingPass() {
  PassRegistration<TestParallelLoopUnrollingPass>();
}
} // namespace test
} // namespace mlir
