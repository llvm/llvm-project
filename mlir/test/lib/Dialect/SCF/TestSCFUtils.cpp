//===- TestSCFUtils.cpp --- Pass to test independent SCF dialect utils ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test SCF dialect utils.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {
class TestSCFForUtilsPass
    : public PassWrapper<TestSCFForUtilsPass, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const final { return "test-scf-for-utils"; }
  StringRef getDescription() const final { return "test scf.for utils"; }
  explicit TestSCFForUtilsPass() = default;

  void runOnOperation() override {
    FuncOp func = getOperation();
    SmallVector<scf::ForOp, 4> toErase;

    func.walk([&](Operation *fakeRead) {
      if (fakeRead->getName().getStringRef() != "fake_read")
        return;
      auto *fakeCompute = fakeRead->getResult(0).use_begin()->getOwner();
      auto *fakeWrite = fakeCompute->getResult(0).use_begin()->getOwner();
      auto loop = fakeRead->getParentOfType<scf::ForOp>();

      OpBuilder b(loop);
      (void)loop.moveOutOfLoop({fakeRead});
      fakeWrite->moveAfter(loop);
      auto newLoop = cloneWithNewYields(b, loop, fakeRead->getResult(0),
                                        fakeCompute->getResult(0));
      fakeCompute->getResult(0).replaceAllUsesWith(
          newLoop.getResults().take_back()[0]);
      toErase.push_back(loop);
    });
    for (auto loop : llvm::reverse(toErase))
      loop.erase();
  }
};

class TestSCFIfUtilsPass
    : public PassWrapper<TestSCFIfUtilsPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "test-scf-if-utils"; }
  StringRef getDescription() const final { return "test scf.if utils"; }
  explicit TestSCFIfUtilsPass() = default;

  void runOnOperation() override {
    int count = 0;
    getOperation().walk([&](scf::IfOp ifOp) {
      auto strCount = std::to_string(count++);
      FuncOp thenFn, elseFn;
      OpBuilder b(ifOp);
      IRRewriter rewriter(b);
      if (failed(outlineIfOp(rewriter, ifOp, &thenFn,
                             std::string("outlined_then") + strCount, &elseFn,
                             std::string("outlined_else") + strCount))) {
        this->signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};

static const StringLiteral kTestPipeliningLoopMarker =
    "__test_pipelining_loop__";
static const StringLiteral kTestPipeliningStageMarker =
    "__test_pipelining_stage__";
/// Marker to express the order in which operations should be after pipelining.
static const StringLiteral kTestPipeliningOpOrderMarker =
    "__test_pipelining_op_order__";

static const StringLiteral kTestPipeliningAnnotationPart =
    "__test_pipelining_part";
static const StringLiteral kTestPipeliningAnnotationIteration =
    "__test_pipelining_iteration";

class TestSCFPipeliningPass
    : public PassWrapper<TestSCFPipeliningPass, OperationPass<FuncOp>> {
public:
  TestSCFPipeliningPass() = default;
  TestSCFPipeliningPass(const TestSCFPipeliningPass &) {}
  StringRef getArgument() const final { return "test-scf-pipelining"; }
  StringRef getDescription() const final { return "test scf.forOp pipelining"; }

  Option<bool> annotatePipeline{
      *this, "annotate",
      llvm::cl::desc("Annote operations during loop pipelining transformation"),
      llvm::cl::init(false)};

  static void
  getSchedule(scf::ForOp forOp,
              std::vector<std::pair<Operation *, unsigned>> &schedule) {
    if (!forOp->hasAttr(kTestPipeliningLoopMarker))
      return;
    schedule.resize(forOp.getBody()->getOperations().size() - 1);
    forOp.walk([&schedule](Operation *op) {
      auto attrStage =
          op->getAttrOfType<IntegerAttr>(kTestPipeliningStageMarker);
      auto attrCycle =
          op->getAttrOfType<IntegerAttr>(kTestPipeliningOpOrderMarker);
      if (attrCycle && attrStage) {
        schedule[attrCycle.getInt()] =
            std::make_pair(op, unsigned(attrStage.getInt()));
      }
    });
  }

  static void annotate(Operation *op,
                       mlir::scf::PipeliningOption::PipelinerPart part,
                       unsigned iteration) {
    OpBuilder b(op);
    switch (part) {
    case mlir::scf::PipeliningOption::PipelinerPart::Prologue:
      op->setAttr(kTestPipeliningAnnotationPart, b.getStringAttr("prologue"));
      break;
    case mlir::scf::PipeliningOption::PipelinerPart::Kernel:
      op->setAttr(kTestPipeliningAnnotationPart, b.getStringAttr("kernel"));
      break;
    case mlir::scf::PipeliningOption::PipelinerPart::Epilogue:
      op->setAttr(kTestPipeliningAnnotationPart, b.getStringAttr("epilogue"));
      break;
    }
    op->setAttr(kTestPipeliningAnnotationIteration,
                b.getI32IntegerAttr(iteration));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, StandardOpsDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::scf::PipeliningOption options;
    options.getScheduleFn = getSchedule;
    if (annotatePipeline)
      options.annotateFn = annotate;
    scf::populateSCFLoopPipeliningPatterns(patterns, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    getOperation().walk([](Operation *op) {
      // Clean up the markers.
      op->removeAttr(kTestPipeliningStageMarker);
      op->removeAttr(kTestPipeliningOpOrderMarker);
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestSCFUtilsPass() {
  PassRegistration<TestSCFForUtilsPass>();
  PassRegistration<TestSCFIfUtilsPass>();
  PassRegistration<TestSCFPipeliningPass>();
}
} // namespace test
} // namespace mlir
