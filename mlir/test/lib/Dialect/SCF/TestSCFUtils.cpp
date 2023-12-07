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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestSCFForUtilsPass
    : public PassWrapper<TestSCFForUtilsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSCFForUtilsPass)

  StringRef getArgument() const final { return "test-scf-for-utils"; }
  StringRef getDescription() const final { return "test scf.for utils"; }
  explicit TestSCFForUtilsPass() = default;
  TestSCFForUtilsPass(const TestSCFForUtilsPass &pass) : PassWrapper(pass) {}

  Option<bool> testReplaceWithNewYields{
      *this, "test-replace-with-new-yields",
      llvm::cl::desc("Test replacing a loop with a new loop that returns new "
                     "additional yield values"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<scf::ForOp, 4> toErase;

    if (testReplaceWithNewYields) {
      func.walk([&](scf::ForOp forOp) {
        if (forOp.getNumResults() == 0)
          return;
        auto newInitValues = forOp.getInitArgs();
        if (newInitValues.empty())
          return;
        NewYieldValueFn fn = [&](OpBuilder &b, Location loc,
                                 ArrayRef<BlockArgument> newBBArgs) {
          Block *block = newBBArgs.front().getOwner();
          SmallVector<Value> newYieldValues;
          for (auto yieldVal :
               cast<scf::YieldOp>(block->getTerminator()).getResults()) {
            newYieldValues.push_back(
                b.create<arith::AddFOp>(loc, yieldVal, yieldVal));
          }
          return newYieldValues;
        };
        OpBuilder b(forOp);
        replaceLoopWithNewYields(b, forOp, newInitValues, fn);
      });
    }
  }
};

struct TestSCFIfUtilsPass
    : public PassWrapper<TestSCFIfUtilsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSCFIfUtilsPass)

  StringRef getArgument() const final { return "test-scf-if-utils"; }
  StringRef getDescription() const final { return "test scf.if utils"; }
  explicit TestSCFIfUtilsPass() = default;

  void runOnOperation() override {
    int count = 0;
    getOperation().walk([&](scf::IfOp ifOp) {
      auto strCount = std::to_string(count++);
      func::FuncOp thenFn, elseFn;
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
/// Marker to express the order in which operations should be after
/// pipelining.
static const StringLiteral kTestPipeliningOpOrderMarker =
    "__test_pipelining_op_order__";

static const StringLiteral kTestPipeliningAnnotationPart =
    "__test_pipelining_part";
static const StringLiteral kTestPipeliningAnnotationIteration =
    "__test_pipelining_iteration";

struct TestSCFPipeliningPass
    : public PassWrapper<TestSCFPipeliningPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSCFPipeliningPass)

  TestSCFPipeliningPass() = default;
  TestSCFPipeliningPass(const TestSCFPipeliningPass &) {}
  StringRef getArgument() const final { return "test-scf-pipelining"; }
  StringRef getDescription() const final { return "test scf.forOp pipelining"; }

  Option<bool> annotatePipeline{
      *this, "annotate",
      llvm::cl::desc("Annote operations during loop pipelining transformation"),
      llvm::cl::init(false)};

  Option<bool> noEpiloguePeeling{
      *this, "no-epilogue-peeling",
      llvm::cl::desc("Use predicates instead of peeling the epilogue."),
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
        // TODO: Index can be out-of-bounds if ops of the loop body disappear
        // due to folding.
        schedule[attrCycle.getInt()] =
            std::make_pair(op, unsigned(attrStage.getInt()));
      }
    });
  }

  /// Helper to generate "predicated" version of `op`. For simplicity we just
  /// wrap the operation in a scf.ifOp operation.
  static Operation *predicateOp(RewriterBase &rewriter, Operation *op,
                                Value pred) {
    Location loc = op->getLoc();
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, op->getResultTypes(), pred, true);
    // True branch.
    op->moveBefore(&ifOp.getThenRegion().front(),
                   ifOp.getThenRegion().front().begin());
    rewriter.setInsertionPointAfter(op);
    if (op->getNumResults() > 0)
      rewriter.create<scf::YieldOp>(loc, op->getResults());
    // False branch.
    rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    SmallVector<Value> elseYieldOperands;
    elseYieldOperands.reserve(ifOp.getNumResults());
    if (auto viewOp = dyn_cast<memref::SubViewOp>(op)) {
      // For sub-views, just clone the op.
      // NOTE: This is okay in the test because we use dynamic memref sizes, so
      // the verifier will not complain. Otherwise, we may create a logically
      // out-of-bounds view and a different technique should be used.
      Operation *opClone = rewriter.clone(*op);
      elseYieldOperands.append(opClone->result_begin(), opClone->result_end());
    } else {
      // Default to assuming constant numeric values.
      for (Type type : op->getResultTypes()) {
        elseYieldOperands.push_back(rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(type)));
      }
    }
    if (op->getNumResults() > 0)
      rewriter.create<scf::YieldOp>(loc, elseYieldOperands);
    return ifOp.getOperation();
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
    registry.insert<arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::scf::PipeliningOption options;
    options.getScheduleFn = getSchedule;
    if (annotatePipeline)
      options.annotateFn = annotate;
    if (noEpiloguePeeling) {
      options.peelEpilogue = false;
      options.predicateFn = predicateOp;
    }
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
