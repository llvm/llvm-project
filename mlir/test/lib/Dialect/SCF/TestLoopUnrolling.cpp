//===-------- TestLoopUnrolling.cpp --- loop unrolling test pass ----------===//
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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static unsigned getNestingDepth(Operation *op) {
  Operation *currOp = op;
  unsigned depth = 0;
  while ((currOp = currOp->getParentOp())) {
    if (isa<scf::ForOp>(currOp))
      depth++;
  }
  return depth;
}

class TestLoopUnrollingPass
    : public PassWrapper<TestLoopUnrollingPass, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const final { return "test-loop-unrolling"; }
  StringRef getDescription() const final {
    return "Tests loop unrolling transformation";
  }
  TestLoopUnrollingPass() = default;
  TestLoopUnrollingPass(const TestLoopUnrollingPass &) {}
  explicit TestLoopUnrollingPass(uint64_t unrollFactorParam,
                                 unsigned loopDepthParam,
                                 bool annotateLoopParam) {
    unrollFactor = unrollFactorParam;
    loopDepth = loopDepthParam;
    annotateLoop = annotateLoopParam;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, StandardOpsDialect>();
  }

  void runOnOperation() override {
    FuncOp func = getOperation();
    SmallVector<scf::ForOp, 4> loops;
    func.walk([&](scf::ForOp forOp) {
      if (getNestingDepth(forOp) == loopDepth)
        loops.push_back(forOp);
    });
    auto annotateFn = [this](unsigned i, Operation *op, OpBuilder b) {
      if (annotateLoop) {
        op->setAttr("unrolled_iteration", b.getUI32IntegerAttr(i));
      }
    };
    for (auto loop : loops)
      (void)loopUnrollByFactor(loop, unrollFactor, annotateFn);
  }
  Option<uint64_t> unrollFactor{*this, "unroll-factor",
                                llvm::cl::desc("Loop unroll factor."),
                                llvm::cl::init(1)};
  Option<bool> annotateLoop{*this, "annotate",
                            llvm::cl::desc("Annotate unrolled iterations."),
                            llvm::cl::init(false)};
  Option<bool> unrollUpToFactor{*this, "unroll-up-to-factor",
                                llvm::cl::desc("Loop unroll up to factor."),
                                llvm::cl::init(false)};
  Option<unsigned> loopDepth{*this, "loop-depth", llvm::cl::desc("Loop depth."),
                             llvm::cl::init(0)};
};
} // namespace

namespace mlir {
namespace test {
void registerTestLoopUnrollingPass() {
  PassRegistration<TestLoopUnrollingPass>();
}
} // namespace test
} // namespace mlir
