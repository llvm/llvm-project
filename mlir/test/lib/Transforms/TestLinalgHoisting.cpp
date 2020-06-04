//===- TestLinalgHoisting.cpp - Test Linalg hoisting functions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Linalg hoisting functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestLinalgHoisting
    : public PassWrapper<TestLinalgHoisting, FunctionPass> {
  TestLinalgHoisting() = default;
  TestLinalgHoisting(const TestLinalgHoisting &pass) {}

  void runOnFunction() override;

  Option<bool> testHoistViewAllocs{
      *this, "test-hoist-view-allocs",
      llvm::cl::desc("Test hoisting alloc used by view"),
      llvm::cl::init(false)};
};
} // end anonymous namespace

void TestLinalgHoisting::runOnFunction() {
  if (testHoistViewAllocs) {
    hoistViewAllocOps(getFunction());
    return;
  }
}

namespace mlir {
void registerTestLinalgHoisting() {
  PassRegistration<TestLinalgHoisting> testTestLinalgHoistingPass(
      "test-linalg-hoisting", "Test Linalg hoisting functions.");
}
} // namespace mlir
