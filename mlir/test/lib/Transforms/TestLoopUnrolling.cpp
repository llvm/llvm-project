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

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

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
    : public PassWrapper<TestLoopUnrollingPass, FunctionPass> {
public:
  TestLoopUnrollingPass() = default;
  TestLoopUnrollingPass(const TestLoopUnrollingPass &) {}
  explicit TestLoopUnrollingPass(uint64_t unrollFactorParam,
                                 unsigned loopDepthParam) {
    unrollFactor = unrollFactorParam;
    loopDepth = loopDepthParam;
  }

  void runOnFunction() override {
    FuncOp func = getFunction();
    SmallVector<scf::ForOp, 4> loops;
    func.walk([&](scf::ForOp forOp) {
      if (getNestingDepth(forOp) == loopDepth)
        loops.push_back(forOp);
    });
    for (auto loop : loops) {
      loopUnrollByFactor(loop, unrollFactor);
    }
  }
  Option<uint64_t> unrollFactor{*this, "unroll-factor",
                                llvm::cl::desc("Loop unroll factor."),
                                llvm::cl::init(1)};
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
  PassRegistration<TestLoopUnrollingPass>(
      "test-loop-unrolling", "Tests loop unrolling transformation");
}
} // namespace test
} // namespace mlir
