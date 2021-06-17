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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }
  StringRef getArgument() const final { return "test-linalg-hoisting"; }
  StringRef getDescription() const final {
    return "Test Linalg hoisting functions.";
  }

  void runOnFunction() override;

  Option<bool> testHoistRedundantTransfers{
      *this, "test-hoist-redundant-transfers",
      llvm::cl::desc("Test hoisting transfer_read/transfer_write pairs"),
      llvm::cl::init(false)};
};
} // end anonymous namespace

void TestLinalgHoisting::runOnFunction() {
  if (testHoistRedundantTransfers) {
    hoistRedundantVectorTransfers(getFunction());
    hoistRedundantVectorTransfersOnTensor(getFunction());
    return;
  }
}

namespace mlir {
namespace test {
void registerTestLinalgHoisting() { PassRegistration<TestLinalgHoisting>(); }
} // namespace test
} // namespace mlir
