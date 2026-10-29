//===- TestTypeOffset.cpp - Test Ptr type offsets ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestPtrSetTypeOffset
    : public PassWrapper<TestPtrSetTypeOffset, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPtrSetTypeOffset)

  TestPtrSetTypeOffset() = default;
  TestPtrSetTypeOffset(const TestPtrSetTypeOffset &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "test-ptr-set-type-offset"; }
  StringRef getDescription() const final {
    return "Attach the queried type offset to each ptr.type_offset operation";
  }

  void runOnOperation() override {
    Builder builder(&getContext());
    getOperation()->walk([&](ptr::TypeOffsetOp op) {
      llvm::TypeSize size = useExplicitLayout
                                ? op.getTypeSize(DataLayout::closest(op))
                                : op.getTypeSize();
      op->setAttr("type_offset", builder.getIndexAttr(size.getKnownMinValue()));
      if (size.isScalable())
        op->setAttr("type_offset_scalable", builder.getUnitAttr());
    });
  }

  Option<bool> useExplicitLayout{
      *this, "use-explicit-layout",
      llvm::cl::desc("Pass the enclosing data layout explicitly to the query"),
      llvm::cl::init(false)};
};
} // namespace

namespace mlir::test {
void registerTestPtrSetTypeOffset() {
  PassRegistration<TestPtrSetTypeOffset>();
}
} // namespace mlir::test
