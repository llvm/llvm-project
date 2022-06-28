//===- TestLinalgElementwiseFusion.cpp - Test Linalg elementwise fusion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing fusion of elementwise operations in
// Linalg, mainly linalg options.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

static void addOperands(Operation *op, SetVector<Value> &operandSet) {
  if (!op)
    return;
  TypeSwitch<Operation *, void>(op)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
        SmallVector<Value> inputOperands = linalgOp.getInputOperands();
        operandSet.insert(inputOperands.begin(), inputOperands.end());
      })
      .Default([&](Operation *operation) {
        operandSet.insert(operation->operand_begin(), operation->operand_end());
      });
}

template <int limit = 3>
static bool setFusedOpOperandLimit(const OpResult &producer,
                                   const OpOperand &consumer) {
  SetVector<Value> fusedOpOperands;
  if (producer.getOwner()->getNumResults() != 1)
    return false;
  addOperands(consumer.getOwner(), fusedOpOperands);
  fusedOpOperands.remove(producer);
  addOperands(producer.getOwner(), fusedOpOperands);
  return fusedOpOperands.size() <= limit;
}

namespace {
struct TestLinalgElementwiseFusion
    : public PassWrapper<TestLinalgElementwiseFusion,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgElementwiseFusion)

  TestLinalgElementwiseFusion() = default;
  TestLinalgElementwiseFusion(const TestLinalgElementwiseFusion &pass)
      : PassWrapper(pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    tensor::TensorDialect>();
  }
  StringRef getArgument() const final {
    return "test-linalg-elementwise-fusion-patterns";
  }
  StringRef getDescription() const final {
    return "Test Linalg element wise operation fusion patterns";
  }

  Option<bool> fuseGenericOps{
      *this, "fuse-generic-ops",
      llvm::cl::desc("Test fusion of generic operations."),
      llvm::cl::init(false)};

  Option<bool> fuseWithReshapeByExpansion{
      *this, "fuse-with-reshape-by-expansion",
      llvm::cl::desc(
          "Test fusion of generic operations with reshape by expansion"),
      llvm::cl::init(false)};

  Option<bool> controlFuseByExpansion{
      *this, "control-fusion-by-expansion",
      llvm::cl::desc(
          "Test controlling fusion of reshape with generic op by expansion"),
      llvm::cl::init(false)};

  Option<bool> fuseWithReshapeByCollapsing{
      *this, "fuse-with-reshape-by-collapsing",
      llvm::cl::desc("Test linalg expand_shape -> generic fusion patterns that "
                     "collapse the iteration space of the consumer"),
      llvm::cl::init(false)};

  Option<bool> fuseWithReshapeByCollapsingWithControlFn{
      *this, "fuse-with-reshape-by-collapsing-control",
      llvm::cl::desc("Test controlling the linalg expand_shape -> generic "
                     "fusion patterns that "
                     "collapse the iteration space of the consumer"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    MLIRContext *context = &this->getContext();
    func::FuncOp funcOp = this->getOperation();

    if (fuseGenericOps) {
      RewritePatternSet fusionPatterns(context);
      linalg::populateElementwiseOpsFusionPatterns(fusionPatterns,
                                                   setFusedOpOperandLimit<4>);

      (void)applyPatternsAndFoldGreedily(funcOp.getBody(),
                                         std::move(fusionPatterns));
      return;
    }

    if (fuseWithReshapeByExpansion) {
      RewritePatternSet fusionPatterns(context);
      linalg::populateFoldReshapeOpsByExpansionPatterns(
          fusionPatterns, [](const OpResult & /*producer*/,
                             OpOperand & /*consumer*/) { return true; });
      if (failed(applyPatternsAndFoldGreedily(funcOp.getBody(),
                                              std::move(fusionPatterns))))
        return signalPassFailure();
      return;
    }

    if (controlFuseByExpansion) {
      RewritePatternSet fusionPatterns(context);

      linalg::ControlFusionFn controlReshapeFusionFn =
          [](const OpResult &producer, OpOperand &consumer) {
            if (auto collapseOp =
                    producer.getDefiningOp<tensor::CollapseShapeOp>()) {
              if (!collapseOp.getSrc().getDefiningOp<linalg::LinalgOp>()) {
                return false;
              }
            }
            if (auto expandOp =
                    dyn_cast<tensor::ExpandShapeOp>(consumer.getOwner())) {
              if (expandOp->hasOneUse()) {
                OpOperand &use = *expandOp->getUses().begin();
                auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
                if (linalgOp && linalgOp.isOutputTensor(&use))
                  return true;
              }
              return false;
            }
            return true;
          };

      linalg::populateFoldReshapeOpsByExpansionPatterns(fusionPatterns,
                                                        controlReshapeFusionFn);
      (void)applyPatternsAndFoldGreedily(funcOp.getBody(),
                                         std::move(fusionPatterns));
      return;
    }

    if (fuseWithReshapeByCollapsing) {
      RewritePatternSet patterns(context);
      linalg::populateFoldReshapeOpsByCollapsingPatterns(
          patterns, [](const OpResult & /*producer*/,
                       OpOperand & /*consumer*/) { return true; });
      (void)applyPatternsAndFoldGreedily(funcOp.getBody(), std::move(patterns));
    }

    if (fuseWithReshapeByCollapsingWithControlFn) {
      RewritePatternSet patterns(context);
      linalg::ControlFusionFn controlFn = [](const OpResult &producer,
                                             OpOperand &consumer) -> bool {
        if (isa<tensor::ExpandShapeOp>(producer.getDefiningOp())) {
          // Skip fusing the first operand.
          return consumer.getOperandNumber();
        }
        return true;
      };
      linalg::populateFoldReshapeOpsByCollapsingPatterns(patterns, controlFn);
      (void)applyPatternsAndFoldGreedily(funcOp.getBody(), std::move(patterns));
    }
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestLinalgElementwiseFusion() {
  PassRegistration<TestLinalgElementwiseFusion>();
}
} // namespace test
} // namespace mlir
