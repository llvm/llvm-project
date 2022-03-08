//===- TestMemRefBoundCheck.cpp - Test out of bound access checks ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to check memref accesses for out of bound
// accesses.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memref-bound-check"

using namespace mlir;

namespace {

/// Checks for out of bound memref access subscripts..
struct TestMemRefBoundCheck
    : public PassWrapper<TestMemRefBoundCheck, OperationPass<>> {
  StringRef getArgument() const final { return "test-memref-bound-check"; }
  StringRef getDescription() const final {
    return "Check memref access bounds";
  }
  void runOnOperation() override;
};

} // namespace

void TestMemRefBoundCheck::runOnOperation() {
  getOperation()->walk([](Operation *opInst) {
    TypeSwitch<Operation *>(opInst)
        .Case<AffineReadOpInterface, AffineWriteOpInterface>(
            [](auto op) { (void)boundCheckLoadOrStoreOp(op); });

    // TODO: do this for DMA ops as well.
  });
}

namespace mlir {
namespace test {
void registerMemRefBoundCheck() { PassRegistration<TestMemRefBoundCheck>(); }
} // namespace test
} // namespace mlir
