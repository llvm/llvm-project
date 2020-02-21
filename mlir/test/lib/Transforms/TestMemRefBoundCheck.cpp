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

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Passes.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memref-bound-check"

using namespace mlir;

namespace {

/// Checks for out of bound memef access subscripts..
struct TestMemRefBoundCheck : public FunctionPass<TestMemRefBoundCheck> {
  void runOnFunction() override;
};

} // end anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createTestMemRefBoundCheckPass() {
  return std::make_unique<TestMemRefBoundCheck>();
}

void TestMemRefBoundCheck::runOnFunction() {
  getFunction().walk([](Operation *opInst) {
    TypeSwitch<Operation *>(opInst).Case<AffineLoadOp, AffineStoreOp>(
        [](auto op) { boundCheckLoadOrStoreOp(op); });

    // TODO(bondhugula): do this for DMA ops as well.
  });
}

namespace mlir {
void registerMemRefBoundCheck() {
  PassRegistration<TestMemRefBoundCheck>(
      "test-memref-bound-check", "Check memref access bounds in a Function");
}
} // namespace mlir
