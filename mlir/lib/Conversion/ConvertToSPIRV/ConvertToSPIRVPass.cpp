//===- ConvertToSPIRVPass.cpp - MLIR SPIR-V Conversion --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToSPIRV/ConvertToSPIRVPass.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/UBToSPIRV/UBToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

#define DEBUG_TYPE "convert-to-spirv"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// A pass to perform the SPIR-V conversion.
struct ConvertToSPIRVPass final
    : impl::ConvertToSPIRVPassBase<ConvertToSPIRVPass> {
  using ConvertToSPIRVPassBase::ConvertToSPIRVPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    if (runSignatureConversion) {
      // Unroll vectors in function signatures to native vector size.
      RewritePatternSet patterns(context);
      populateFuncOpVectorRewritePatterns(patterns);
      populateReturnOpVectorRewritePatterns(patterns);
      GreedyRewriteConfig config;
      config.strictMode = GreedyRewriteStrictness::ExistingOps;
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config)))
        return signalPassFailure();
    }

    if (runVectorUnrolling) {
      // Fold transpose ops if possible as we cannot unroll it later.
      {
        RewritePatternSet patterns(context);
        vector::TransposeOp::getCanonicalizationPatterns(patterns, context);
        if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
          return signalPassFailure();
        }
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

      // Convert transpose ops into extract and insert pairs, in preparation
      // of further transformations to canonicalize/cancel.
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

        // Pull in casting way leading one dims to allow cancelling some
        // read/write ops.
        vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
        vector::ReductionOp::getCanonicalizationPatterns(patterns, context);

        // Decompose different rank insert_strided_slice and n-D
        // extract_slided_slice.
        vector::populateVectorInsertExtractStridedSliceDecompositionPatterns(
            patterns);
        vector::ExtractOp::getCanonicalizationPatterns(patterns, context);

        // Trimming leading unit dims may generate broadcast/shape_cast ops.
        // Clean them up.
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

    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);
    SPIRVTypeConverter typeConverter(targetAttr);
    RewritePatternSet patterns(context);
    ScfToSPIRVContext scfToSPIRVContext;

    // Populate patterns for each dialect.
    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    arith::populateArithToSPIRVPatterns(typeConverter, patterns);
    populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);
    populateFuncToSPIRVPatterns(typeConverter, patterns);
    index::populateIndexToSPIRVPatterns(typeConverter, patterns);
    populateVectorToSPIRVPatterns(typeConverter, patterns);
    populateSCFToSPIRVPatterns(typeConverter, scfToSPIRVContext, patterns);
    ub::populateUBToSPIRVConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
