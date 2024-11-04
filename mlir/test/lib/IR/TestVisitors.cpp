//===- TestIRVisitors.cpp - Pass to test the IR visitors ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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

  llvm::outs() << "Op reverse post-order visits"
               << "\n";
  op->walk<WalkOrder::PostOrder, ReverseIterator>(opPure);
  llvm::outs() << "Block reverse post-order visits"
               << "\n";
  op->walk<WalkOrder::PostOrder, ReverseIterator>(blockPure);
  llvm::outs() << "Region reverse post-order visits"
               << "\n";
  op->walk<WalkOrder::PostOrder, ReverseIterator>(regionPure);

  // This test case tests "NoGraphRegions = true", so start the walk with
  // functions.
  op->walk([&](FunctionOpInterface funcOp) {
    llvm::outs() << "Op forward dominance post-order visits"
                 << "\n";
    funcOp->walk<WalkOrder::PostOrder,
                 ForwardDominanceIterator</*NoGraphRegions=*/true>>(opPure);
    llvm::outs() << "Block forward dominance post-order visits"
                 << "\n";
    funcOp->walk<WalkOrder::PostOrder,
                 ForwardDominanceIterator</*NoGraphRegions=*/true>>(blockPure);
    llvm::outs() << "Region forward dominance post-order visits"
                 << "\n";
    funcOp->walk<WalkOrder::PostOrder,
                 ForwardDominanceIterator</*NoGraphRegions=*/true>>(regionPure);

    llvm::outs() << "Op reverse dominance post-order visits"
                 << "\n";
    funcOp->walk<WalkOrder::PostOrder,
                 ReverseDominanceIterator</*NoGraphRegions=*/true>>(opPure);
    llvm::outs() << "Block reverse dominance post-order visits"
                 << "\n";
    funcOp->walk<WalkOrder::PostOrder,
                 ReverseDominanceIterator</*NoGraphRegions=*/true>>(blockPure);
    llvm::outs() << "Region reverse dominance post-order visits"
                 << "\n";
    funcOp->walk<WalkOrder::PostOrder,
                 ReverseDominanceIterator</*NoGraphRegions=*/true>>(regionPure);
  });
}

/// Tests erasure callbacks that skip the walk.
static void testSkipErasureCallbacks(Operation *op) {
  auto skipOpErasure = [](Operation *op) {
    // Do not erase module and module children operations. Otherwise, there
    // wouldn't be too much to test in pre-order.
    if (isa<ModuleOp>(op) || isa<ModuleOp>(op->getParentOp()))
      return WalkResult::advance();

    llvm::outs() << "Erasing ";
    printOperation(op);
    llvm::outs() << "\n";
    op->dropAllUses();
    op->erase();
    return WalkResult::skip();
  };
  auto skipBlockErasure = [](Block *block) {
    // Do not erase module and module children blocks. Otherwise there wouldn't
    // be too much to test in pre-order.
    Operation *parentOp = block->getParentOp();
    if (isa<ModuleOp>(parentOp) || isa<ModuleOp>(parentOp->getParentOp()))
      return WalkResult::advance();

    if (block->use_empty()) {
      llvm::outs() << "Erasing ";
      printBlock(block);
      llvm::outs() << "\n";
      block->erase();
      return WalkResult::skip();
    } else {
      llvm::outs() << "Cannot erase ";
      printBlock(block);
      llvm::outs() << ", still has uses\n";
      return WalkResult::advance();
    }
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
    if (block->use_empty()) {
      llvm::outs() << "Erasing ";
      printBlock(block);
      llvm::outs() << "\n";
      block->erase();
    } else {
      llvm::outs() << "Cannot erase ";
      printBlock(block);
      llvm::outs() << ", still has uses\n";
    }
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

/// Invoke region/block walks on regions/blocks.
static void testBlockAndRegionWalkers(Operation *op) {
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

  llvm::outs() << "Invoke block pre-order visits on blocks\n";
  op->walk([&](Operation *op) {
    if (!op->hasAttr("walk_blocks"))
      return;
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        block.walk<WalkOrder::PreOrder>(blockPure);
      }
    }
  });

  llvm::outs() << "Invoke block post-order visits on blocks\n";
  op->walk([&](Operation *op) {
    if (!op->hasAttr("walk_blocks"))
      return;
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        block.walk<WalkOrder::PostOrder>(blockPure);
      }
    }
  });

  llvm::outs() << "Invoke region pre-order visits on region\n";
  op->walk([&](Operation *op) {
    if (!op->hasAttr("walk_regions"))
      return;
    for (Region &region : op->getRegions()) {
      region.walk<WalkOrder::PreOrder>(regionPure);
    }
  });

  llvm::outs() << "Invoke region post-order visits on region\n";
  op->walk([&](Operation *op) {
    if (!op->hasAttr("walk_regions"))
      return;
    for (Region &region : op->getRegions()) {
      region.walk<WalkOrder::PostOrder>(regionPure);
    }
  });
}

namespace {
/// This pass exercises the different configurations of the IR visitors.
struct TestIRVisitorsPass
    : public PassWrapper<TestIRVisitorsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIRVisitorsPass)

  StringRef getArgument() const final { return "test-ir-visitors"; }
  StringRef getDescription() const final { return "Test various visitors."; }
  void runOnOperation() override {
    Operation *op = getOperation();
    testPureCallbacks(op);
    testBlockAndRegionWalkers(op);
    testSkipErasureCallbacks(op);
    testNoSkipErasureCallbacks(op);
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestIRVisitorsPass() { PassRegistration<TestIRVisitorsPass>(); }
} // namespace test
} // namespace mlir
