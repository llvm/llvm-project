//===- TestProcessMultiIndexOpLowering.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestMultiIndexOpLoweringPass
    : public PassWrapper<TestMultiIndexOpLoweringPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMultiIndexOpLoweringPass)

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mesh::MeshDialect>();
    mesh::processMultiIndexOpLoweringRegisterDialects(registry);
  }
  StringRef getArgument() const final {
    return "test-mesh-process-multi-index-op-lowering";
  }
  StringRef getDescription() const final {
    return "Test lowering of mesh.process_multi_index op.";
  }
};
} // namespace

void TestMultiIndexOpLoweringPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  SymbolTableCollection symbolTableCollection;
  mesh::processMultiIndexOpLoweringPopulatePatterns(patterns,
                                                    symbolTableCollection);
  LogicalResult status =
      applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  (void)status;
  assert(succeeded(status) && "applyPatternsAndFoldGreedily failed.");
}

namespace mlir {
namespace test {
void registerTestMultiIndexOpLoweringPass() {
  PassRegistration<TestMultiIndexOpLoweringPass>();
}
} // namespace test
} // namespace mlir
