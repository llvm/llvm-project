//===- TestTopologicalSort.cpp - Pass to test topological sort analysis ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

using namespace mlir;

namespace {
struct TestTopologicalSortAnalysisPass
    : public PassWrapper<TestTopologicalSortAnalysisPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTopologicalSortAnalysisPass)

  StringRef getArgument() const final {
    return "test-topological-sort-analysis";
  }
  StringRef getDescription() const final {
    return "Test topological sorting of ops";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    OpBuilder builder(op->getContext());

    op->walk([&](Operation *root) {
      if (!root->hasAttr("root"))
        return WalkResult::advance();

      assert(root->getNumRegions() == 1 && root->getRegion(0).hasOneBlock() &&
             "expected one block");
      Block *block = &root->getRegion(0).front();
      SmallVector<Operation *> selectedOps;
      block->walk([&](Operation *op) {
        if (op->hasAttr("selected"))
          selectedOps.push_back(op);
      });

      computeTopologicalSorting(block, selectedOps);
      for (const auto &it : llvm::enumerate(selectedOps))
        it.value()->setAttr("pos", builder.getIndexAttr(it.index()));

      return WalkResult::advance();
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestTopologicalSortAnalysisPass() {
  PassRegistration<TestTopologicalSortAnalysisPass>();
}
} // namespace test
} // namespace mlir
