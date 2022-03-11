//===- TestLiveness.cpp - Test liveness construction and information ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and resolving liveness
// information.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestLivenessPass
    : public PassWrapper<TestLivenessPass, InterfacePass<SymbolOpInterface>> {
  StringRef getArgument() const final { return "test-print-liveness"; }
  StringRef getDescription() const final {
    return "Print the contents of a constructed liveness information.";
  }
  void runOnOperation() override {
    llvm::errs() << "Testing : " << getOperation().getName() << "\n";
    getAnalysis<Liveness>().print(llvm::errs());
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestLivenessPass() { PassRegistration<TestLivenessPass>(); }
} // namespace test
} // namespace mlir
