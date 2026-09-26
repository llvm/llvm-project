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
      Attribute defaultMemorySpace = layout.getDefaultMemorySpace();
      Attribute allocaMemorySpace = layout.getAllocaMemorySpace();
      Attribute manglingMode = layout.getManglingMode();
      Attribute programMemorySpace = layout.getProgramMemorySpace();
      Attribute globalMemorySpace = layout.getGlobalMemorySpace();
      uint64_t stackAlignment = layout.getStackAlignment();
      Attribute functionPointerAlignment = layout.getFunctionPointerAlignment();
      Attribute legalIntWidths = layout.getLegalIntWidths();

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
           builder.getNamedAttr("default_memory_space",
                                defaultMemorySpace == Attribute()
                                    ? builder.getUI32IntegerAttr(0)
                                    : defaultMemorySpace),
           builder.getNamedAttr("alloca_memory_space",
                                allocaMemorySpace == Attribute()
                                    ? builder.getUI32IntegerAttr(0)
                                    : allocaMemorySpace),
           builder.getNamedAttr("mangling_mode", manglingMode == Attribute()
                                                     ? builder.getStringAttr("")
                                                     : manglingMode),
           builder.getNamedAttr("program_memory_space",
                                programMemorySpace == Attribute()
                                    ? builder.getUI32IntegerAttr(0)
                                    : programMemorySpace),
           builder.getNamedAttr("global_memory_space",
                                globalMemorySpace == Attribute()
                                    ? builder.getUI32IntegerAttr(0)
                                    : globalMemorySpace),
           builder.getNamedAttr("stack_alignment",
                                builder.getIndexAttr(stackAlignment)),
           builder.getNamedAttr("function_pointer_alignment",
                                functionPointerAlignment == Attribute()
                                    ? FunctionPointerAlignmentAttr::get(
                                          builder.getContext(), 0,
                                          /*function_dependent=*/false)
                                    : functionPointerAlignment),
           builder.getNamedAttr("legal_int_widths",
                                legalIntWidths == Attribute()
                                    ? builder.getDenseI32ArrayAttr({})
                                    : legalIntWidths)

          });
    });
  }
};

struct TestDLTIQueryOpInterface
    : public PassWrapper<TestDLTIQueryOpInterface, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDLTIQueryOpInterface)

  StringRef getArgument() const final { return "test-dlti-query-op-interface"; }
  StringRef getDescription() const final {
    return "Test the DLTI query operation interface";
  }

  void runOnOperation() override {
    WalkResult result = getOperation()->walk([&](Operation *op) {
      auto action =
          op->getDiscardableAttrOfType<StringAttr>("test.dlti_action");
      if (!action)
        return WalkResult::advance();

      auto queryOp = dyn_cast<DLTIQueryOpInterface>(op);
      if (!queryOp) {
        op->emitError("expected DLTIQueryOpInterface");
        return WalkResult::interrupt();
      }

      Builder builder(op->getContext());
      auto key = [&](StringRef value) { return builder.getStringAttr(value); };
      if (action.getValue() == "update") {
        if (failed(queryOp.setDlti(key("inserted"),
                                   builder.getI32IntegerAttr(1))) ||
            failed(queryOp.setDlti(key("replaced"),
                                   builder.getI32IntegerAttr(2))) ||
            failed(queryOp.setDlti(builder.getI32Type(),
                                   builder.getI32IntegerAttr(32))) ||
            failed(queryOp.setDlti(key("removed"), Attribute())) ||
            failed(queryOp.setDlti(key("absent"), Attribute()))) {
          op->emitError("failed to update DLTI map");
          return WalkResult::interrupt();
        }
      } else if (action.getValue() == "clear") {
        if (failed(queryOp.setDlti(key("only"), Attribute())) ||
            failed(queryOp.setDlti(key("absent"), Attribute()))) {
          op->emitError("failed to clear DLTI map");
          return WalkResult::interrupt();
        }
      } else if (action.getValue() == "fail") {
        if (succeeded(
                queryOp.setDlti(key("new"), builder.getI32IntegerAttr(1)))) {
          op->emitError("unexpectedly updated immutable DLTI representation");
          return WalkResult::interrupt();
        }
        op->setDiscardableAttr("test.set_dlti_failed", builder.getUnitAttr());
      } else if (action.getValue() == "invalid") {
        Attribute value = builder.getI32IntegerAttr(1);
        if (succeeded(queryOp.setDlti(key(""), value)) ||
            succeeded(queryOp.setDlti(DataLayoutEntryKey(), value)) ||
            succeeded(queryOp.setDlti(key(""), Attribute()))) {
          op->emitError("unexpectedly accepted an invalid DLTI key");
          return WalkResult::interrupt();
        }
      }
      op->removeDiscardableAttr("test.dlti_action");
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestDataLayoutQuery() {
  PassRegistration<TestDataLayoutQuery>();
  PassRegistration<TestDLTIQueryOpInterface>();
}
} // namespace test
} // namespace mlir
