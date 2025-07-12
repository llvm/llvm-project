//===- TestOneShotRootBufferzation.cpp - Bufferization Test -----*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotRootBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestOneShotRootBufferizePass
    : public PassWrapper<TestOneShotRootBufferizePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOneShotRootBufferizePass)

  TestOneShotRootBufferizePass() = default;
  TestOneShotRootBufferizePass(const TestOneShotRootBufferizePass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }
  StringRef getArgument() const final { return "test-one-shot-root-bufferize"; }
  StringRef getDescription() const final {
    return "Module pass to test One Shot Root Bufferization";
  }

  void runOnOperation() override {

    llvm::errs() << "Running TestOneShotRootBufferize on: "
                 << getOperation()->getName() << "\n";
    bufferization::OneShotBufferizationOptions opt;

    opt.bufferizeFunctionBoundaries = true;
    bufferization::BufferizationState bufferizationState;

    if (failed(bufferization::runOneShotRootBufferize(getOperation(), opt,
                                                      bufferizationState)))
      signalPassFailure();
  }
};
} // namespace

namespace mlir::test {
void registerTestOneShotRootBufferizePass() {
  PassRegistration<TestOneShotRootBufferizePass>();
}
} // namespace mlir::test
