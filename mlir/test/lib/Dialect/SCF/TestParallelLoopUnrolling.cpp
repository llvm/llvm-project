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
    SmallVector<scf::ParallelOp, 4> loops;
    getOperation()->walk([&](scf::ParallelOp parLoop) {
      if (getNestingDepth(parLoop) == loopDepth)
        loops.push_back(parLoop);
    });
    auto annotateFn = [this](unsigned i, Operation *op, OpBuilder b) {
      if (annotateLoop) {
        op->setAttr("unrolled_iteration", b.getUI32IntegerAttr(i));
      }
    };
    PatternRewriter rewriter(getOperation()->getContext());
    for (auto loop : loops) {
      (void)parallelLoopUnrollByFactors(loop, unrollFactors, rewriter,
                                        annotateFn);
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
};
} // namespace

namespace mlir {
namespace test {
void registerTestParallelLoopUnrollingPass() {
  PassRegistration<TestParallelLoopUnrollingPass>();
}
} // namespace test
} // namespace mlir
