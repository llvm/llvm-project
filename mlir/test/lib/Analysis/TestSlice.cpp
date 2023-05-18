//===------------- TestSlice.cpp - Test slice related analisis ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

static const StringLiteral kOrderMarker = "__test_sort_original_idx__";

namespace {

struct TestTopologicalSortPass
    : public PassWrapper<TestTopologicalSortPass,
                         InterfacePass<SymbolOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTopologicalSortPass)

  StringRef getArgument() const final { return "test-print-topological-sort"; }
  StringRef getDescription() const final {
    return "Print operations in topological order";
  }
  void runOnOperation() override {
    std::map<int, Operation *> ops;
    getOperation().walk([&ops](Operation *op) {
      if (auto originalOrderAttr = op->getAttrOfType<IntegerAttr>(kOrderMarker))
        ops[originalOrderAttr.getInt()] = op;
    });
    SetVector<Operation *> sortedOp;
    for (auto op : ops)
      sortedOp.insert(op.second);
    sortedOp = topologicalSort(sortedOp);
    llvm::errs() << "Testing : " << getOperation().getName() << "\n";
    for (Operation *op : sortedOp) {
      op->print(llvm::errs());
      llvm::errs() << "\n";
    }
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
