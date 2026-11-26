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
#include "llvm/ADT/SmallVectorExtras.h"

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
    // Combines the values of the unrolled iterations with a single cast, to
    // show which values are combined.
    auto combineWithCast = [](RewriterBase &rewriter, scf::ReduceOp reduceOp,
                              ArrayRef<SmallVector<Value>> iterReduced) {
      SmallVector<Value> reduced;
      for (auto [idx, value] : llvm::enumerate(iterReduced.front())) {
        SmallVector<Value> inputs = llvm::map_to_vector(
            iterReduced, [&](ArrayRef<Value> iter) { return iter[idx]; });
        reduced.push_back(
            UnrealizedConversionCastOp::create(rewriter, reduceOp.getLoc(),
                                               value.getType(), inputs)
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
      // Number the original ops to show which op each clone is mapped to.
      DenseMap<Operation *, unsigned> srcOpIndices;
      for (Operation &op : loop.getBody()->without_terminator())
        srcOpIndices.try_emplace(&op, srcOpIndices.size());
      IRMapping clonedToSrcOpsMap;
      if (failed(parallelLoopUnrollByFactors(loop, unrollFactors, rewriter,
                                             annotateFn, &clonedToSrcOpsMap,
                                             combineFn)) ||
          !mapClonedOps)
        continue;
      for (auto [cloned, src] : clonedToSrcOpsMap.getOperationMap())
        if (auto it = srcOpIndices.find(src); it != srcOpIndices.end())
          cloned->setDiscardableAttr("cloned_from",
                                     rewriter.getUI32IntegerAttr(it->second));
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
      llvm::cl::desc("Annotate cloned ops with the original op they map to."),
      llvm::cl::init(false)};
  Option<bool> combineCast{
      *this, "combine-with-cast",
      llvm::cl::desc("Combine the reduced values of the unrolled iterations "
                     "with a cast."),
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
