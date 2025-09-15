//===- XeGPUVectorLinearize.cpp - Linearizes n-D vectors to 1-D vectors
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUVECTORLINEARIZE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-vector-linearize"

using namespace mlir;

namespace {
struct XeGPUVectorLinearizePass final
    : public xegpu::impl::XeGPUVectorLinearizeBase<XeGPUVectorLinearizePass> {
  void runOnOperation() override {
    // vector.broadcast and vector.gather requires progressive lowering
    {
      mlir::RewritePatternSet patterns(&getContext());
      mlir::vector::populateVectorBroadcastLoweringPatterns(patterns);
      mlir::vector::populateVectorGatherLoweringPatterns(patterns);
      mlir::vector::populateVectorGatherToConditionalLoadPatterns(patterns);
      // vector.transpose lowering
      // Shuffle16x16 will fallback to Shuffle1D for non 16x16 sizes.
      mlir::vector::populateVectorTransposeLoweringPatterns(
          patterns, mlir::vector::VectorTransposeLowering::Shuffle16x16);
      (void)mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    // Unroll load store from <<MxN> to M <1xN> load/stores and then linearize
    {
      mlir::RewritePatternSet patterns(&getContext());
      mlir::vector::UnrollVectorOptions vectorOptions;
      vectorOptions.setNativeShapeFn(
          [](mlir::Operation *op) -> std::optional<mlir::SmallVector<int64_t>> {
            auto extractVectorType =
                [](mlir::Operation *op) -> mlir::VectorType {
              if (auto loadOp = mlir::dyn_cast<mlir::vector::LoadOp>(op))
                return mlir::dyn_cast<mlir::VectorType>(
                    loadOp.getResult().getType());
              if (auto storeOp = mlir::dyn_cast<mlir::vector::StoreOp>(op))
                return mlir::dyn_cast<mlir::VectorType>(
                    storeOp.getValueToStore().getType());
              return nullptr;
            };

            auto vecType = extractVectorType(op);
            if (!vecType)
              return std::nullopt;

            auto shape = vecType.getShape();
            if (shape.size() != 2)
              return std::nullopt;

            return mlir::SmallVector<int64_t>{1, shape[1]};
          });
      mlir::vector::populateVectorUnrollPatterns(patterns, vectorOptions);
      (void)mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    // Use vector linearization patterns
    {
      mlir::MLIRContext &context = getContext();
      mlir::TypeConverter converter;
      mlir::RewritePatternSet patterns(&context);
      mlir::ConversionTarget target(context);
      mlir::vector::populateForVectorLinearize(converter, target);
      mlir::vector::populateVectorLinearizeBasePatterns(converter, target,
                                                        patterns);
      mlir::vector::populateVectorLinearizeShuffleLikeOpsPatterns(
          converter, target, patterns);
      mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
          converter, patterns, target);
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }
  }
};
} // namespace
