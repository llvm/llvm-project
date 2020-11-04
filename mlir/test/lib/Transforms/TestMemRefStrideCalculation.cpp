//===- TestMemRefStrideCalculation.cpp - Pass to test strides computation--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct TestMemRefStrideCalculation
    : public PassWrapper<TestMemRefStrideCalculation, FunctionPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

/// Traverse AllocOp and compute strides of each MemRefType independently.
void TestMemRefStrideCalculation::runOnFunction() {
  llvm::outs() << "Testing: " << getFunction().getName() << "\n";
  getFunction().walk([&](AllocOp allocOp) {
    auto memrefType = allocOp.getResult().getType().cast<MemRefType>();
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (failed(getStridesAndOffset(memrefType, strides, offset))) {
      llvm::outs() << "MemRefType " << memrefType << " cannot be converted to "
                   << "strided form\n";
      return;
    }
    llvm::outs() << "MemRefType offset: ";
    if (offset == MemRefType::getDynamicStrideOrOffset())
      llvm::outs() << "?";
    else
      llvm::outs() << offset;
    llvm::outs() << " strides: ";
    llvm::interleaveComma(strides, llvm::outs(), [&](int64_t v) {
      if (v == MemRefType::getDynamicStrideOrOffset())
        llvm::outs() << "?";
      else
        llvm::outs() << v;
    });
    llvm::outs() << "\n";
  });
  llvm::outs().flush();
}

namespace mlir {
namespace test {
void registerTestMemRefStrideCalculation() {
  PassRegistration<TestMemRefStrideCalculation> pass(
      "test-memref-stride-calculation", "Test operation constant folding");
}
} // namespace test
} // namespace mlir
