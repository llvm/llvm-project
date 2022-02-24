//===- TestComposeSubView.cpp - Test composed subviews --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestMultiBufferingPass
    : public PassWrapper<TestMultiBufferingPass, OperationPass<FuncOp>> {
  TestMultiBufferingPass() = default;
  TestMultiBufferingPass(const TestMultiBufferingPass &pass)
      : PassWrapper(pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }
  StringRef getArgument() const final { return "test-multi-buffering"; }
  StringRef getDescription() const final {
    return "Test multi buffering transformation";
  }
  void runOnOperation() override;
  Option<unsigned> multiplier{
      *this, "multiplier",
      llvm::cl::desc(
          "Decide how many versions of the buffer should be created,"),
      llvm::cl::init(2)};
};

void TestMultiBufferingPass::runOnOperation() {
  SmallVector<memref::AllocOp> allocs;
  getOperation().walk(
      [&allocs](memref::AllocOp alloc) { allocs.push_back(alloc); });
  for (memref::AllocOp alloc : allocs)
    (void)multiBuffer(alloc, multiplier);
}
} // namespace

namespace mlir {
namespace test {
void registerTestMultiBuffering() {
  PassRegistration<TestMultiBufferingPass>();
}
} // namespace test
} // namespace mlir
