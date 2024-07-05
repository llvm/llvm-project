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
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include <memory>

#define DEBUG_TYPE "convert-to-spirv"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Vector Lowering
//===----------------------------------------------------------------------===//

int getComputeVectorSize(int64_t size) {
  for (int i : {4, 3, 2}) {
    if (size % i == 0)
      return i;
  }
  return 1;
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::MultiDimReductionOp op) {
  // Unroll all reduction dimensions by size 1 for vector.multi_reduction.
  VectorType srcVectorType = op.getSourceVectorType();
  auto nativeSize = llvm::to_vector(srcVectorType.getShape());
  auto dims = op.getReductionDims().getAsValueRange<IntegerAttr>();
  for (const auto &dimAttr : dims) {
    nativeSize[dimAttr.getZExtValue()] = 1;
  }
  return nativeSize;
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::ReductionOp op) {
  VectorType srcVectorType = op.getSourceVectorType();
  assert(srcVectorType.getRank() == 1); // Guaranteed by semantics
  int64_t vectorSize = getComputeVectorSize(srcVectorType.getDimSize(0));
  return {vectorSize};
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::TransposeOp op) {
  VectorType vectorType = op.getResultVectorType();
  SmallVector<int64_t> nativeSize(vectorType.getRank(), 1);
  nativeSize.back() = getComputeVectorSize(vectorType.getShape().back());
  return nativeSize;
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::GatherOp op) {
  VectorType vectorType = op.getVectorType();
  SmallVector<int64_t> nativeSize(vectorType.getRank(), 1);
  nativeSize.back() = getComputeVectorSize(vectorType.getShape().back());
  return nativeSize;
}

std::optional<SmallVector<int64_t>> getNativeVectorShape(Operation *op) {
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = getComputeVectorSize(vecType.getShape().back());
      return nativeSize;
    }
  }

  return TypeSwitch<Operation *, std::optional<SmallVector<int64_t>>>(op)
      .Case<vector::MultiDimReductionOp, vector::ReductionOp,
            vector::TransposeOp, vector::GatherOp>(
          [](auto typedOp) { return getNativeVectorShapeImpl(typedOp); })
      .Default([](Operation *) { return std::nullopt; });
}

namespace {

/// A pass to perform the SPIR-V conversion.
struct ConvertToSPIRVPass final
    : impl::ConvertToSPIRVPassBase<ConvertToSPIRVPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    // Unroll vectors in function inputs to native vector size.
    {
      llvm::errs() << "Start unrolling function inputs\n";
      RewritePatternSet patterns(context);
      populateFuncOpVectorRewritePatterns(patterns);
      GreedyRewriteConfig config;
      config.strictMode = GreedyRewriteStrictness::ExistingOps;
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config)))
        return signalPassFailure();
      llvm::errs() << "Finish unrolling function inputs\n";
    }

    SPIRVTypeConverter typeConverter(targetAttr);

    // Unroll vectors to native vector size.
    {
      RewritePatternSet patterns(context);
      auto options = vector::UnrollVectorOptions().setNativeShapeFn(
          [=](auto op) { return getNativeVectorShape(op); });
      populateVectorUnrollPatterns(patterns, options);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
        return signalPassFailure();
    }

    llvm::errs() << "After unrolling vectors to native vector size\n";
    op->dump();

    // Next run canonicalization to cast away leading size-1 dimensions.
    {
      RewritePatternSet patterns(context);

      // We need to pull in casting way leading one dims to allow cancelling
      // some read/write ops.
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);

      // We may have vector.insert_strided_slice inserting 1-D native vectors
      // into n-D larger vectors with the above. Break that down too. This is a
      // companion transformation of unrolling.
      vector::populateVectorInsertExtractStridedSliceDecompositionPatterns(
          patterns);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);

      // Trimming leading unit dims may generate broadcast/shape_cast ops. Clean
      // them up.
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);

      vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);

      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
        return signalPassFailure();
    }

    llvm::errs() << "After running canonicalization to cast away leading size-1 dimensions\n";
    op->dump();

    // Convert vector.extract_strided_slice into a chain of vector.extract and
    // then a chain of vector.insert ops. This helps to cancel with previous
    // vector.insert/extract ops, especially for fP16 cases where we have
    // mismatched vector size for transfer and compute.
    {
      RewritePatternSet patterns(context);
      vector::populateVectorExtractStridedSliceToExtractInsertChainPatterns(
          patterns, [](vector::ExtractStridedSliceOp op) {
            return op.getSourceVectorType().getNumElements() > 4;
          });
      vector::InsertOp::getCanonicalizationPatterns(patterns, context);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
        return signalPassFailure();
    }

    llvm::errs() << "After converting vector.extract_strided_slice into a chain of vector.extract and then a chain of vector.insert ops\n";
    op->dump();

    // Run all sorts of canonicalization patterns to clean up again.
    {
      RewritePatternSet patterns(context);
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::InsertOp::getCanonicalizationPatterns(patterns, context);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
      vector::ReductionOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
        return signalPassFailure();
    }

    llvm::errs() << "After running canonicalization patterns to clean up again\n";
    op->dump();

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
