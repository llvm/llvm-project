//===- TestDataLayoutQuery.cpp - Test Data Layout Queries -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestOps.h"
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
      if (op->getDiscardableAttr("size"))
        return;

      const DataLayout &layout = layouts.getAbove(op);
      llvm::TypeSize size = layout.getTypeSize(op.getType());
      llvm::TypeSize bitsize = layout.getTypeSizeInBits(op.getType());
      uint64_t alignment = layout.getTypeABIAlignment(op.getType());
      uint64_t preferred = layout.getTypePreferredAlignment(op.getType());
      uint64_t index = layout.getTypeIndexBitwidth(op.getType()).value_or(0);
      Attribute endianness = layout.getEndianness();
      Attribute allocaMemorySpace = layout.getAllocaMemorySpace();
      Attribute programMemorySpace = layout.getProgramMemorySpace();
      Attribute globalMemorySpace = layout.getGlobalMemorySpace();
      uint64_t stackAlignment = layout.getStackAlignment();

      auto convertTypeSizeToAttr = [&](llvm::TypeSize typeSize) -> Attribute {
        if (!typeSize.isScalable())
          return builder.getIndexAttr(typeSize);

        return builder.getDictionaryAttr({
            builder.getNamedAttr("scalable", builder.getUnitAttr()),
            builder.getNamedAttr(
                "minimal_size",
                builder.getIndexAttr(typeSize.getKnownMinValue())),
        });
      };

      op->setAttrs(
          {builder.getNamedAttr("size", convertTypeSizeToAttr(size)),
           builder.getNamedAttr("bitsize", convertTypeSizeToAttr(bitsize)),
           builder.getNamedAttr("alignment", builder.getIndexAttr(alignment)),
           builder.getNamedAttr("preferred", builder.getIndexAttr(preferred)),
           builder.getNamedAttr("index", builder.getIndexAttr(index)),
           builder.getNamedAttr("endianness", endianness == Attribute()
                                                  ? builder.getStringAttr("")
                                                  : endianness),
           builder.getNamedAttr("alloca_memory_space",
                                allocaMemorySpace == Attribute()
                                    ? builder.getUI32IntegerAttr(0)
                                    : allocaMemorySpace),
           builder.getNamedAttr("program_memory_space",
                                programMemorySpace == Attribute()
                                    ? builder.getUI32IntegerAttr(0)
                                    : programMemorySpace),
           builder.getNamedAttr("global_memory_space",
                                globalMemorySpace == Attribute()
                                    ? builder.getUI32IntegerAttr(0)
                                    : globalMemorySpace),
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
