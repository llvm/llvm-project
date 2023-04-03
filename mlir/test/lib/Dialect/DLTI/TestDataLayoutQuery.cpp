//===- TestDataLayoutQuery.cpp - Test Data Layout Queries -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// A pass that finds "test.data_layout_query" operations and attaches to them
/// attributes containing the results of data layout queries for operation
/// result types.
struct TestDataLayoutQuery
    : public PassWrapper<TestDataLayoutQuery, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDataLayoutQuery)

  StringRef getArgument() const final { return "test-data-layout-query"; }
  StringRef getDescription() const final { return "Test data layout queries"; }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Builder builder(func.getContext());
    const DataLayoutAnalysis &layouts = getAnalysis<DataLayoutAnalysis>();

    func.walk([&](test::DataLayoutQueryOp op) {
      // Skip the ops with already processed in a deeper call.
      if (op->getAttr("size"))
        return;

      const DataLayout &layout = layouts.getAbove(op);
      unsigned size = layout.getTypeSize(op.getType());
      unsigned bitsize = layout.getTypeSizeInBits(op.getType());
      unsigned alignment = layout.getTypeABIAlignment(op.getType());
      unsigned preferred = layout.getTypePreferredAlignment(op.getType());
      Attribute allocaMemorySpace = layout.getAllocaMemorySpace();
      unsigned stackAlignment = layout.getStackAlignment();
      op->setAttrs(
          {builder.getNamedAttr("size", builder.getIndexAttr(size)),
           builder.getNamedAttr("bitsize", builder.getIndexAttr(bitsize)),
           builder.getNamedAttr("alignment", builder.getIndexAttr(alignment)),
           builder.getNamedAttr("preferred", builder.getIndexAttr(preferred)),
           builder.getNamedAttr("alloca_memory_space",
                                allocaMemorySpace == Attribute()
                                    ? builder.getUI32IntegerAttr(0)
                                    : allocaMemorySpace),
           builder.getNamedAttr("stack_alignment",
                                builder.getIndexAttr(stackAlignment))});
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestDataLayoutQuery() { PassRegistration<TestDataLayoutQuery>(); }
} // namespace test
} // namespace mlir
