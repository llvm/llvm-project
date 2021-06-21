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

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {

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
    : public PassWrapper<TestLinalgElementwiseFusion, FunctionPass> {
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

  void runOnFunction() override {
    MLIRContext *context = &this->getContext();
    FuncOp funcOp = this->getFunction();
    RewritePatternSet fusionPatterns(context);

    linalg::populateElementwiseOpsFusionPatterns(
        fusionPatterns,
        linalg::LinalgElementwiseFusionOptions()
            .setControlElementwiseOpsFusionFn(setFusedOpOperandLimit<4>));

    (void)applyPatternsAndFoldGreedily(funcOp.getBody(),
                                       std::move(fusionPatterns));
  }
};

struct TestPushExpandingReshape
    : public PassWrapper<TestPushExpandingReshape, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, linalg::LinalgDialect, tensor::TensorDialect>();
  }
  StringRef getArgument() const final { return "test-linalg-push-reshape"; }
  StringRef getDescription() const final {
    return "Test Linalg reshape push patterns";
  }

  void runOnFunction() override {
    MLIRContext *context = &this->getContext();
    FuncOp funcOp = this->getFunction();
    RewritePatternSet patterns(context);
    linalg::populatePushReshapeOpsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp.getBody(), std::move(patterns));
  }
};
} // namespace

namespace test {
void registerTestLinalgElementwiseFusion() {
  PassRegistration<TestLinalgElementwiseFusion>();
}

void registerTestPushExpandingReshape() {
  PassRegistration<TestPushExpandingReshape>();
}
} // namespace test

} // namespace mlir
