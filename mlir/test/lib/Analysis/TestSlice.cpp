//===- TestSlice.cpp - Test slice related analisis ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceWalk.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

static const StringLiteral kToSortMark = "test_to_sort";
static const StringLiteral kOrderIndex = "test_sort_index";

namespace {

struct TestTopologicalSortPass
    : public PassWrapper<TestTopologicalSortPass,
                         InterfacePass<SymbolOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTopologicalSortPass)

  StringRef getArgument() const final { return "test-print-topological-sort"; }
  StringRef getDescription() const final {
    return "Sorts operations topologically and attaches attributes with their "
           "corresponding index in the ordering to them";
  }
  void runOnOperation() override {
    SetVector<Operation *> toSort;
    getOperation().walk([&](Operation *op) {
      if (op->hasAttrOfType<UnitAttr>(kToSortMark))
        toSort.insert(op);
    });

    auto i32Type = IntegerType::get(&getContext(), 32);
    SetVector<Operation *> sortedOps = topologicalSort(toSort);
    for (auto [index, op] : llvm::enumerate(sortedOps))
      op->setAttr(kOrderIndex, IntegerAttr::get(i32Type, index));
  }
};

/// Pass to test getControlFlowPredecessors from SliceWalk.
struct TestControlFlowPredecessorsPass
    : public PassWrapper<TestControlFlowPredecessorsPass,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestControlFlowPredecessorsPass)

  StringRef getArgument() const final {
    return "test-control-flow-predecessors";
  }
  StringRef getDescription() const final {
    return "Test getControlFlowPredecessors from SliceWalk.";
  }

  void runOnOperation() override {
    FunctionOpInterface func = getOperation();
    llvm::errs() << "Control flow predecessors for '" << func.getNameAttr()
                 << "':\n";
    func->walk([](Operation *op) {
      if (!isa<RegionBranchOpInterface>(op))
        return;
      for (OpResult result : op->getResults()) {
        auto predecessors = mlir::getControlFlowPredecessors(result);
        llvm::errs() << "  '" << op->getName() << "': result #"
                     << result.getResultNumber() << ": ";
        if (!predecessors)
          llvm::errs() << "no predecessors\n";
        else
          llvm::errs() << predecessors->size() << " predecessor(s)\n";
      }
      for (Region &region : op->getRegions()) {
        if (region.empty())
          continue;
        for (BlockArgument arg : region.front().getArguments()) {
          auto predecessors = mlir::getControlFlowPredecessors(arg);
          llvm::errs() << "  '" << op->getName() << "': region #"
                       << region.getRegionNumber() << " block arg #"
                       << arg.getArgNumber() << ": ";
          if (!predecessors)
            llvm::errs() << "no predecessors\n";
          else
            llvm::errs() << predecessors->size() << " predecessor(s)\n";
        }
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestSliceAnalysisPass() {
  PassRegistration<TestTopologicalSortPass>();
  PassRegistration<TestControlFlowPredecessorsPass>();
}
} // namespace test
} // namespace mlir
