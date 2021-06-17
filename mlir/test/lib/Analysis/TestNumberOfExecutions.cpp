//===- TestNumberOfExecutions.cpp - Test number of executions analysis ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and resolving number of
// executions information.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/NumberOfExecutions.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestNumberOfBlockExecutionsPass
    : public PassWrapper<TestNumberOfBlockExecutionsPass, FunctionPass> {
  StringRef getArgument() const final {
    return "test-print-number-of-block-executions";
  }
  StringRef getDescription() const final {
    return "Print the contents of a constructed number of executions analysis "
           "for "
           "all blocks.";
  }
  void runOnFunction() override {
    llvm::errs() << "Number of executions: " << getFunction().getName() << "\n";
    getAnalysis<NumberOfExecutions>().printBlockExecutions(
        llvm::errs(), &getFunction().getBody());
  }
};

struct TestNumberOfOperationExecutionsPass
    : public PassWrapper<TestNumberOfOperationExecutionsPass, FunctionPass> {
  StringRef getArgument() const final {
    return "test-print-number-of-operation-executions";
  }
  StringRef getDescription() const final {
    return "Print the contents of a constructed number of executions analysis "
           "for "
           "all operations.";
  }
  void runOnFunction() override {
    llvm::errs() << "Number of executions: " << getFunction().getName() << "\n";
    getAnalysis<NumberOfExecutions>().printOperationExecutions(
        llvm::errs(), &getFunction().getBody());
  }
};

} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestNumberOfBlockExecutionsPass() {
  PassRegistration<TestNumberOfBlockExecutionsPass>();
}

void registerTestNumberOfOperationExecutionsPass() {
  PassRegistration<TestNumberOfOperationExecutionsPass>();
}
} // namespace test
} // namespace mlir
