//===- TestTensorCopyInsertion.cpp - Bufferization Analysis -----*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This pass runs One-Shot Analysis and inserts copies for all OpOperands that
/// were decided to bufferize out-of-place. After running this pass, a
/// bufferization can write to buffers directly (without making copies) and no
/// longer has to care about potential read-after-write conflicts.
///
/// Note: By default, all newly inserted tensor copies/allocs (i.e., newly
/// created `bufferization.alloc_tensor` ops) that do not escape block are
/// annotated with `escape = false`. If `create-allocs` is unset, all newly
/// inserted tensor copies/allocs are annotated with `escape = true`. In that
/// case, they are not getting deallocated when bufferizing the IR.
struct TestTensorCopyInsertionPass
    : public PassWrapper<TestTensorCopyInsertionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTensorCopyInsertionPass)

  TestTensorCopyInsertionPass() = default;
  TestTensorCopyInsertionPass(const TestTensorCopyInsertionPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }
  StringRef getArgument() const final { return "test-tensor-copy-insertion"; }
  StringRef getDescription() const final {
    return "Module pass to test Tensor Copy Insertion";
  }

  void runOnOperation() override {
    bufferization::OneShotBufferizationOptions options;
    options.allowReturnAllocsFromLoops = allowReturnAllocsFromLoops;
    options.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
    if (mustInferMemorySpace) {
      options.defaultMemorySpaceFn =
          [](TensorType t) -> std::optional<Attribute> { return std::nullopt; };
    }
    if (failed(bufferization::insertTensorCopies(getOperation(), options)))
      signalPassFailure();
  }

  Option<bool> allowReturnAllocsFromLoops{
      *this, "allow-return-allocs-from-loops",
      llvm::cl::desc("Allows returning/yielding new allocations from a loop."),
      llvm::cl::init(false)};
  Option<bool> bufferizeFunctionBoundaries{
      *this, "bufferize-function-boundaries",
      llvm::cl::desc("Bufferize function boundaries."), llvm::cl::init(false)};
  Option<bool> mustInferMemorySpace{
      *this, "must-infer-memory-space",
      llvm::cl::desc(
          "The memory space of an memref types must always be inferred. If "
          "unset, a default memory space of 0 is used otherwise."),
      llvm::cl::init(false)};
};
} // namespace

namespace mlir::test {
void registerTestTensorCopyInsertionPass() {
  PassRegistration<TestTensorCopyInsertionPass>();
}
} // namespace mlir::test
