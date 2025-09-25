//===- TestUpliftWhileToFor.cpp - while to for loop uplifting test pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to test transforms SCF.WhileOp's into SCF.ForOp's.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TestSCFUpliftWhileToFor
    : public PassWrapper<TestSCFUpliftWhileToFor, OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSCFUpliftWhileToFor)

  StringRef getArgument() const final { return "test-scf-uplift-while-to-for"; }

  StringRef getDescription() const final {
    return "test scf while to for uplifting";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    scf::populateUpliftWhileToForPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestSCFUpliftWhileToFor() {
  PassRegistration<TestSCFUpliftWhileToFor>();
}
} // namespace test
} // namespace mlir
