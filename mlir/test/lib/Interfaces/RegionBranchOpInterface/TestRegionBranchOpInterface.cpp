//===- TestBlockInLoop.cpp - Pass to test mlir::blockIsInLoop -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
/// This is a test pass that tests Blocks's isInLoop method by checking if each
/// block in a function is in a loop and outputing if it is
struct PrintRegionBranchOpInterfacePass
    : public PassWrapper<PrintRegionBranchOpInterfacePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintRegionBranchOpInterfacePass)

  StringRef getArgument() const final {
    return "print-region-branch-op-interface";
  }
  StringRef getDescription() const final {
    return "Print control-flow edges represented by "
           "mlir::RegionBranchOpInterface";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    op->walk<WalkOrder::PreOrder>([&](RegionBranchOpInterface branchOp) {
      llvm::outs() << "Found RegionBranchOpInterface operation: "
                   << OpWithFlags(
                          branchOp,
                          OpPrintingFlags().skipRegions().enableDebugInfo())
                   << "\n";
      SmallVector<RegionSuccessor> regions;
      branchOp.getSuccessorRegions(RegionBranchPoint::parent(), regions);
      for (auto &successor : regions) {
        if (successor.isParent()) {
          llvm::outs() << " - Successor is parent\n";
        } else {
          llvm::outs() << " - Successor is region #"
                       << successor.getSuccessor()->getRegionNumber() << "\n";
        }
      }
      if (auto breakingControlFlowOp =
              dyn_cast<HasBreakingControlFlowOpInterface>(
                  branchOp.getOperation())) {
        SmallVector<Operation *> predecessors;
        llvm::outs() << " - Collecting all nested predecessors\n";
        collectAllNestedPredecessors(breakingControlFlowOp, predecessors);
        llvm::outs() << " - Found " << predecessors.size()
                     << " predecessor(s)\n";
        for (auto &predecessor : predecessors) {
          llvm::outs() << "  - Predecessor is "
                       << OpWithFlags(
                              predecessor,
                              OpPrintingFlags().skipRegions().enableDebugInfo())
                       << "\n";
        }
      }
    });
  }
};

} // namespace

namespace mlir {
void registerRegionBranchOpInterfaceTestPasses() {
  PassRegistration<PrintRegionBranchOpInterfacePass>();
}
} // namespace mlir
