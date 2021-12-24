//===- TestCallGraph.cpp - Test callgraph construction and iteration ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and iterating over a
// callgraph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestCallGraphPass
    : public PassWrapper<TestCallGraphPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "test-print-callgraph"; }
  StringRef getDescription() const final {
    return "Print the contents of a constructed callgraph.";
  }
  void runOnOperation() override {
    llvm::errs() << "Testing : " << getOperation()->getAttr("test.name")
                 << "\n";
    getAnalysis<CallGraph>().print(llvm::errs());
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestCallGraphPass() { PassRegistration<TestCallGraphPass>(); }
} // namespace test
} // namespace mlir
