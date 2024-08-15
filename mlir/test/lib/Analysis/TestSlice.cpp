//===- TestSlice.cpp - Test slice related analisis ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
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

} // namespace

namespace mlir {
namespace test {
void registerTestSliceAnalysisPass() {
  PassRegistration<TestTopologicalSortPass>();
}
} // namespace test
} // namespace mlir
