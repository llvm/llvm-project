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

    WalkResult result = op->walk([&](Operation *root) {
      if (!root->hasAttr("root"))
        return WalkResult::advance();

      SmallVector<Operation *> selectedOps;
      root->walk([&](Operation *selected) {
        if (!selected->hasAttr("selected"))
          return WalkResult::advance();
        if (root->hasAttr("ordered")) {
          // If the root has an "ordered" attribute, we fill the selectedOps
          // vector in a certain order.
          int64_t pos =
              cast<IntegerAttr>(selected->getAttr("selected")).getInt();
          if (pos >= static_cast<int64_t>(selectedOps.size()))
            selectedOps.append(pos + 1 - selectedOps.size(), nullptr);
          selectedOps[pos] = selected;
        } else {
          selectedOps.push_back(selected);
        }
        return WalkResult::advance();
      });

      if (llvm::find(selectedOps, nullptr) != selectedOps.end()) {
        root->emitError("invalid test case: some indices are missing among the "
                        "selected ops");
        return WalkResult::skip();
      }

      if (!computeTopologicalSorting(selectedOps)) {
        root->emitError("could not schedule all ops");
        return WalkResult::skip();
      }

      for (const auto &it : llvm::enumerate(selectedOps))
        it.value()->setAttr("pos", builder.getIndexAttr(it.index()));

      return WalkResult::advance();
    });

    if (result.wasSkipped())
      signalPassFailure();
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
