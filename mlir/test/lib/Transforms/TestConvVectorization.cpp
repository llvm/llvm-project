//===- TestConvVectorization.cpp - Vectorization of Conv ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace vector;

namespace {
/// A pass converting MLIR Linalg ops into Vector ops.
class TestConvVectorization
    : public PassWrapper<TestConvVectorization, OperationPass<ModuleOp>> {
public:
  TestConvVectorization() = default;
  TestConvVectorization(const TestConvVectorization &) {}
  explicit TestConvVectorization(ArrayRef<int64_t> tileSizesParam) {
    tileSizes = tileSizesParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<AffineDialect>();
    registry.insert<StandardOpsDialect>();
  }

  ListOption<int64_t> tileSizes{
      *this, "tile-sizes", llvm::cl::desc("Vectorization sizes."),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};
} // namespace

void TestConvVectorization::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, scf::SCFDialect, StandardOpsDialect,
                         VectorDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ModuleTerminatorOp, ReturnOp>();
  target.addLegalOp<linalg::FillOp, linalg::YieldOp>();

  SmallVector<OwningRewritePatternList, 4> stage1Patterns;
  linalg::populateConvVectorizationPatterns(context, stage1Patterns, tileSizes);
  SmallVector<FrozenRewritePatternList, 4> frozenStage1Patterns;
  llvm::move(stage1Patterns, std::back_inserter(frozenStage1Patterns));

  OwningRewritePatternList stage2Patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  stage2Patterns.insert<linalg::AffineMinSCFCanonicalizationPattern>(context);

  auto stage3Transforms = [](Operation *op) {
    PassManager pm(op->getContext());
    pm.addPass(createLoopInvariantCodeMotionPass());
    if (failed(pm.run(cast<ModuleOp>(op))))
      llvm_unreachable("Unexpected failure in cleanup pass pipeline.");
    op->walk([](FuncOp func) {
      promoteSingleIterationLoops(func);
      linalg::hoistRedundantVectorTransfers(func);
    });
    return success();
  };

  (void)linalg::applyStagedPatterns(module, frozenStage1Patterns,
                                    std::move(stage2Patterns),
                                    stage3Transforms);

  //===--------------------------------------------------------------------===//
  // Post staged patterns transforms
  //===--------------------------------------------------------------------===//

  VectorTransformsOptions vectorTransformsOptions{
      VectorContractLowering::Dot, VectorTransposeLowering::EltWise};

  OwningRewritePatternList vectorTransferPatterns;
  // Pattern is not applied because rank-reducing vector transfer is not yet
  // supported as can be seen in splitFullAndPartialTransferPrecondition,
  // VectorTransforms.cpp
  vectorTransferPatterns.insert<VectorTransferFullPartialRewriter>(
      context, vectorTransformsOptions);
  (void)applyPatternsAndFoldGreedily(module, std::move(vectorTransferPatterns));

  // Programmatic controlled lowering of linalg.copy and linalg.fill.
  PassManager pm(context);
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  if (failed(pm.run(module)))
    llvm_unreachable("Unexpected failure in linalg to loops pass.");

  // Programmatic controlled lowering of vector.contract only.
  OwningRewritePatternList vectorContractLoweringPatterns;
  populateVectorContractLoweringPatterns(vectorContractLoweringPatterns,
                                         context, vectorTransformsOptions);
  (void)applyPatternsAndFoldGreedily(module,
                                     std::move(vectorContractLoweringPatterns));

  // Programmatic controlled lowering of vector.transfer only.
  OwningRewritePatternList vectorToLoopsPatterns;
  populateVectorToSCFConversionPatterns(vectorToLoopsPatterns, context,
                                        VectorTransferToSCFOptions());
  (void)applyPatternsAndFoldGreedily(module, std::move(vectorToLoopsPatterns));

  // Ensure we drop the marker in the end.
  module.walk([](linalg::LinalgOp op) {
    op.removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
  });
}

namespace mlir {
namespace test {
void registerTestConvVectorization() {
  PassRegistration<TestConvVectorization> testTransformPatternsPass(
      "test-conv-vectorization", "Test vectorization of convolutions");
}
} // namespace test
} // namespace mlir
