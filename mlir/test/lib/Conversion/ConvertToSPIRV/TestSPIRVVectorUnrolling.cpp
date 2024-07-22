//===- TestSPIRVVectorUnrolling.cpp - Test signature conversion -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

struct TestSPIRVVectorUnrolling final
    : PassWrapper<TestSPIRVVectorUnrolling, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSPIRVVectorUnrolling)

  StringRef getArgument() const final { return "test-spirv-vector-unrolling"; }

  StringRef getDescription() const final {
    return "Test patterns that unroll vectors to types supported by SPIR-V";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    // Unroll vectors in function signatures to native vector size.
    {
      RewritePatternSet patterns(context);
      populateFuncOpVectorRewritePatterns(patterns);
      populateReturnOpVectorRewritePatterns(patterns);
      GreedyRewriteConfig config;
      config.strictMode = GreedyRewriteStrictness::ExistingOps;
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config)))
        return signalPassFailure();
    }

    // Unroll vectors to native vector size.
    {
      RewritePatternSet patterns(context);
      auto options = vector::UnrollVectorOptions().setNativeShapeFn(
          [=](auto op) { return mlir::spirv::getNativeVectorShape(op); });
      populateVectorUnrollPatterns(patterns, options);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
        return signalPassFailure();
    }

    // Convert transpose ops into extract and insert pairs, in preparation of
    // further transformations to canonicalize/cancel.
    {
      RewritePatternSet patterns(context);
      auto options =
          vector::VectorTransformsOptions().setVectorTransposeLowering(
              vector::VectorTransposeLowering::EltWise);
      vector::populateVectorTransposeLoweringPatterns(patterns, options);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Run canonicalization to cast away leading size-1 dimensions.
    {
      RewritePatternSet patterns(context);

      // We need to pull in casting way leading one dims.
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::ReductionOp::getCanonicalizationPatterns(patterns, context);

      // Decompose different rank insert_strided_slice and n-D
      // extract_slided_slice.
      vector::populateVectorInsertExtractStridedSliceDecompositionPatterns(
          patterns);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);

      // Trimming leading unit dims may generate broadcast/shape_cast ops. Clean
      // them up.
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);

      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
        return signalPassFailure();
    }

    // Run all sorts of canonicalization patterns to clean up again.
    {
      RewritePatternSet patterns(context);
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::InsertOp::getCanonicalizationPatterns(patterns, context);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
      vector::ReductionOp::getCanonicalizationPatterns(patterns, context);
      vector::TransposeOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace test {
void registerTestSPIRVVectorUnrolling() {
  PassRegistration<TestSPIRVVectorUnrolling>();
}
} // namespace test
} // namespace mlir
