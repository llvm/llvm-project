//===- TestTypeOffset.cpp - Test Ptr type offsets ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestPtrSetTypeOffset
    : public PassWrapper<TestPtrSetTypeOffset, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPtrSetTypeOffset)

  StringRef getArgument() const final { return "test-ptr-set-type-offset"; }
  StringRef getDescription() const final {
    return "Attach type offsets to ptr.type_offset operations in func.func";
  }

  void runOnOperation() override {
    Builder builder(&getContext());
    getOperation()->walk([&](func::FuncOp func) {
      DataLayout layout = DataLayout::closest(func);
      func.walk([&](ptr::TypeOffsetOp op) {
        llvm::TypeSize size = op.getTypeSize(layout);
        op->setAttr("type_offset",
                    builder.getIndexAttr(size.getKnownMinValue()));
        if (size.isScalable())
          op->setAttr("type_offset_scalable", builder.getUnitAttr());
      });
    });
  }
};
} // namespace

namespace mlir::test {
void registerTestPtrSetTypeOffset() {
  PassRegistration<TestPtrSetTypeOffset>();
}
} // namespace mlir::test
