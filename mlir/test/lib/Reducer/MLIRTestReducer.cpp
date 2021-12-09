//===- TestReducer.cpp - Test MLIR Reduce ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that reproduces errors based on trivially defined
// patterns. It is used as a buggy optimization pass for the purpose of testing
// the MLIR Reduce tool.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// This pass looks for for the presence of an operation with the name
/// "crashOp" in the input MLIR file and crashes the mlir-opt tool if the
/// operation is found.
struct TestReducer : public PassWrapper<TestReducer, OperationPass<>> {
  StringRef getArgument() const final { return "test-mlir-reducer"; }
  StringRef getDescription() const final {
    return "Tests MLIR Reduce tool by generating failures";
  }
  void runOnOperation() override;
};

} // namespace

void TestReducer::runOnOperation() {
  getOperation()->walk([&](Operation *op) {
    StringRef opName = op->getName().getStringRef();

    if (opName.contains("op_crash")) {
      llvm::errs() << "MLIR Reducer Test generated failure: Found "
                      "\"crashOp\" operation\n";
      exit(1);
    }
  });
}

namespace mlir {
void registerTestReducer() { PassRegistration<TestReducer>(); }
} // namespace mlir
