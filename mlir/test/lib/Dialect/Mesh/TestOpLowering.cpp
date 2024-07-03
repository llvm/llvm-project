//===- TestProcessMultiIndexOpLowering.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Mesh/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TestAllSliceOpLoweringPass
    : public PassWrapper<TestAllSliceOpLoweringPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAllSliceOpLoweringPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    SymbolTableCollection symbolTableCollection;
    mesh::populateAllSliceOpLoweringPatterns(patterns, symbolTableCollection);
    LogicalResult status =
        applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    (void)status;
    assert(succeeded(status) && "applyPatternsAndFoldGreedily failed.");
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    mesh::registerAllSliceOpLoweringDialects(registry);
  }
  StringRef getArgument() const final {
    return "test-mesh-all-slice-op-lowering";
  }
  StringRef getDescription() const final {
    return "Test lowering of all-slice.";
  }
};

struct TestMultiIndexOpLoweringPass
    : public PassWrapper<TestMultiIndexOpLoweringPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMultiIndexOpLoweringPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    SymbolTableCollection symbolTableCollection;
    mesh::populateProcessMultiIndexOpLoweringPatterns(patterns,
                                                      symbolTableCollection);
    LogicalResult status =
        applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    (void)status;
    assert(succeeded(status) && "applyPatternsAndFoldGreedily failed.");
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    mesh::registerProcessMultiIndexOpLoweringDialects(registry);
  }
  StringRef getArgument() const final {
    return "test-mesh-process-multi-index-op-lowering";
  }
  StringRef getDescription() const final {
    return "Test lowering of mesh.process_multi_index op.";
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestOpLoweringPasses() {
  PassRegistration<TestAllSliceOpLoweringPass>();
  PassRegistration<TestMultiIndexOpLoweringPass>();
}
} // namespace test
} // namespace mlir
