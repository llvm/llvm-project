//===- TestDataLayoutQuery.cpp - Test Data Layout Queries -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// A pass that finds "test.data_layout_query" operations and attaches to them
/// attributes containing the results of data layout queries for operation
/// result types.
struct TestDataLayoutQuery
    : public PassWrapper<TestDataLayoutQuery, FunctionPass> {
  void runOnFunction() override {
    FuncOp func = getFunction();
    Builder builder(func.getContext());
    DenseMap<Operation *, DataLayout> layouts;

    func.walk([&](test::DataLayoutQueryOp op) {
      // Skip the ops with already processed in a deeper call.
      if (op->getAttr("size"))
        return;

      auto scope = op->getParentOfType<test::OpWithDataLayoutOp>();
      if (!layouts.count(scope)) {
        layouts.try_emplace(
            scope, scope ? cast<DataLayoutOpInterface>(scope.getOperation())
                         : nullptr);
      }

      const DataLayout &layout = layouts.find(scope)->getSecond();
      unsigned size = layout.getTypeSize(op.getType());
      unsigned alignment = layout.getTypeABIAlignment(op.getType());
      unsigned preferred = layout.getTypePreferredAlignment(op.getType());
      op->setAttrs(
          {builder.getNamedAttr("size", builder.getIndexAttr(size)),
           builder.getNamedAttr("alignment", builder.getIndexAttr(alignment)),
           builder.getNamedAttr("preferred", builder.getIndexAttr(preferred))});
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestDataLayoutQuery() {
  PassRegistration<TestDataLayoutQuery>("test-data-layout-query",
                                        "Test data layout queries");
}
} // namespace test
} // namespace mlir
