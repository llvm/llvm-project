//===- TestOwnershipBasedBufferDeallocation.cpp -----------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
/// This pass runs the ownership based deallocation pass once for `memref.alloc`
/// operations, then lowers the `bufferization.dealloc` operations, and
/// afterwards runs the deallocation pass again for `gpu.alloc` operations and
/// lowers the inserted `bufferization.dealloc` operations again to the
/// corresponding deallocation operations.
struct TestOwnershipBasedBufferDeallocationPass
    : public PassWrapper<TestOwnershipBasedBufferDeallocationPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestOwnershipBasedBufferDeallocationPass)

  TestOwnershipBasedBufferDeallocationPass() = default;
  TestOwnershipBasedBufferDeallocationPass(
      const TestOwnershipBasedBufferDeallocationPass &pass)
      : TestOwnershipBasedBufferDeallocationPass() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    scf::SCFDialect, func::FuncDialect, arith::ArithDialect>();
  }
  StringRef getArgument() const final {
    return "test-ownership-based-buffer-deallocation";
  }
  StringRef getDescription() const final {
    return "Module pass to test the Ownership-based Buffer Deallocation pass";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Build the library function for the lowering of `bufferization.dealloc`.
    OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
    SymbolTable symbolTable(module);
    func::FuncOp helper = bufferization::buildDeallocationLibraryFunction(
        builder, module.getLoc(), symbolTable);

    RewritePatternSet patterns(module->getContext());
    bufferization::populateBufferizationDeallocLoweringPattern(patterns,
                                                               helper);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    WalkResult result = getOperation()->walk([&](FunctionOpInterface funcOp) {
      // Deallocate the `memref.alloc` operations.
      bufferization::DeallocationOptions options;
      options.removeExistingDeallocations = true;
      if (failed(
              bufferization::deallocateBuffersOwnershipBased(funcOp, options)))
        return WalkResult::interrupt();

      // Lower the inserted `bufferization.dealloc` operations.
      ConversionTarget target(getContext());
      target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                             scf::SCFDialect, func::FuncDialect>();
      target.addIllegalOp<bufferization::DeallocOp>();

      if (failed(applyPartialConversion(funcOp, target, frozenPatterns)))
        return WalkResult::interrupt();

      // Deallocate the `gpu.alloc` operations.
      options.isRelevantAllocOp = [](Operation *op) {
        return isa<gpu::GPUDialect>(op->getDialect());
      };
      options.isRelevantDeallocOp = [](Operation *op) {
        return isa<gpu::GPUDialect>(op->getDialect());
      };
      options.getDeallocReplacement =
          [](Operation *op) -> FailureOr<ValueRange> {
        if (auto gpuDealloc = dyn_cast<gpu::DeallocOp>(op)) {
          if (gpuDealloc.getAsyncToken()) {
            OpBuilder builder(op);
            ValueRange token =
                builder
                    .create<gpu::WaitOp>(
                        op->getLoc(),
                        gpu::AsyncTokenType::get(builder.getContext()),
                        ValueRange{})
                    .getResults();
            return token;
          }
          return ValueRange{};
        }
        return failure();
      };
      if (failed(
              bufferization::deallocateBuffersOwnershipBased(funcOp, options)))
        return WalkResult::interrupt();

      // Lower the `bufferization.dealloc` operations inserted in the second
      // deallocation run.
      // TODO: they are currently also lowered to memref.dealloc, we need to
      // add pass options to the lowering pass that allow us to select the
      // dealloc operation to be inserted.
      if (failed(applyPartialConversion(funcOp, target, frozenPatterns)))
        return WalkResult::interrupt();

      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace

namespace mlir::test {
void registerTestOwnershipBasedBufferDeallocationPass() {
  PassRegistration<TestOwnershipBasedBufferDeallocationPass>();
}
} // namespace mlir::test
