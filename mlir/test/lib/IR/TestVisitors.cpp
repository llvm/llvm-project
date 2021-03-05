//===- TestIRVisitors.cpp - Pass to test the IR visitors ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"

using namespace mlir;

static void printRegion(Region *region) {
  llvm::outs() << "region " << region->getRegionNumber() << " from operation '"
               << region->getParentOp()->getName() << "'";
}

static void printBlock(Block *block) {
  llvm::outs() << "block ";
  block->printAsOperand(llvm::outs(), /*printType=*/false);
  llvm::outs() << " from ";
  printRegion(block->getParent());
}

static void printOperation(Operation *op) {
  llvm::outs() << "op '" << op->getName() << "'";
}

/// Tests pure callbacks.
static void testPureCallbacks(Operation *op) {
  auto opPure = [](Operation *op) {
    llvm::outs() << "Visiting ";
    printOperation(op);
    llvm::outs() << "\n";
  };
  auto blockPure = [](Block *block) {
    llvm::outs() << "Visiting ";
    printBlock(block);
    llvm::outs() << "\n";
  };
  auto regionPure = [](Region *region) {
    llvm::outs() << "Visiting ";
    printRegion(region);
    llvm::outs() << "\n";
  };

  llvm::outs() << "Op pre-order visits"
               << "\n";
  op->walk<WalkOrder::PreOrder>(opPure);
  llvm::outs() << "Block pre-order visits"
               << "\n";
  op->walk<WalkOrder::PreOrder>(blockPure);
  llvm::outs() << "Region pre-order visits"
               << "\n";
  op->walk<WalkOrder::PreOrder>(regionPure);

  llvm::outs() << "Op post-order visits"
               << "\n";
  op->walk<WalkOrder::PostOrder>(opPure);
  llvm::outs() << "Block post-order visits"
               << "\n";
  op->walk<WalkOrder::PostOrder>(blockPure);
  llvm::outs() << "Region post-order visits"
               << "\n";
  op->walk<WalkOrder::PostOrder>(regionPure);
}

/// Tests erasure callbacks that skip the walk.
static void testSkipErasureCallbacks(Operation *op) {
  auto skipOpErasure = [](Operation *op) {
    // Do not erase module and function op. Otherwise there wouldn't be too
    // much to test in pre-order.
    if (isa<ModuleOp>(op) || isa<FuncOp>(op))
      return WalkResult::advance();

    llvm::outs() << "Erasing ";
    printOperation(op);
    llvm::outs() << "\n";
    op->dropAllUses();
    op->erase();
    return WalkResult::skip();
  };
  auto skipBlockErasure = [](Block *block) {
    // Do not erase module and function blocks. Otherwise there wouldn't be
    // too much to test in pre-order.
    Operation *parentOp = block->getParentOp();
    if (isa<ModuleOp>(parentOp) || isa<FuncOp>(parentOp))
      return WalkResult::advance();

    llvm::outs() << "Erasing ";
    printBlock(block);
    llvm::outs() << "\n";
    block->erase();
    return WalkResult::skip();
  };

  llvm::outs() << "Op pre-order erasures (skip)"
               << "\n";
  Operation *cloned = op->clone();
  cloned->walk<WalkOrder::PreOrder>(skipOpErasure);
  cloned->erase();

  llvm::outs() << "Block pre-order erasures (skip)"
               << "\n";
  cloned = op->clone();
  cloned->walk<WalkOrder::PreOrder>(skipBlockErasure);
  cloned->erase();

  llvm::outs() << "Op post-order erasures (skip)"
               << "\n";
  cloned = op->clone();
  cloned->walk<WalkOrder::PostOrder>(skipOpErasure);
  cloned->erase();

  llvm::outs() << "Block post-order erasures (skip)"
               << "\n";
  cloned = op->clone();
  cloned->walk<WalkOrder::PostOrder>(skipBlockErasure);
  cloned->erase();
}

/// Tests callbacks that erase the op or block but don't return 'Skip'. This
/// callbacks are only valid in post-order.
static void testNoSkipErasureCallbacks(Operation *op) {
  auto noSkipOpErasure = [](Operation *op) {
    llvm::outs() << "Erasing ";
    printOperation(op);
    llvm::outs() << "\n";
    op->dropAllUses();
    op->erase();
  };
  auto noSkipBlockErasure = [](Block *block) {
    llvm::outs() << "Erasing ";
    printBlock(block);
    llvm::outs() << "\n";
    block->erase();
  };

  llvm::outs() << "Op post-order erasures (no skip)"
               << "\n";
  Operation *cloned = op->clone();
  cloned->walk<WalkOrder::PostOrder>(noSkipOpErasure);

  llvm::outs() << "Block post-order erasures (no skip)"
               << "\n";
  cloned = op->clone();
  cloned->walk<WalkOrder::PostOrder>(noSkipBlockErasure);
  cloned->erase();
}

namespace {
/// This pass exercises the different configurations of the IR visitors.
struct TestIRVisitorsPass
    : public PassWrapper<TestIRVisitorsPass, OperationPass<>> {
  void runOnOperation() override {
    Operation *op = getOperation();
    testPureCallbacks(op);
    testSkipErasureCallbacks(op);
    testNoSkipErasureCallbacks(op);
  }
};
} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestIRVisitorsPass() {
  PassRegistration<TestIRVisitorsPass>("test-ir-visitors",
                                       "Test various visitors.");
}
} // namespace test
} // namespace mlir
