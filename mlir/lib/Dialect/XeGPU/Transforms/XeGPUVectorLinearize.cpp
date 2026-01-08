//===-- XeGPUVectorLinearize.cpp - Linearizes n-D vectors to 1-D vectors --===//
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"

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
      RewritePatternSet patterns(&getContext());
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      vector::populateVectorGatherLoweringPatterns(patterns);
      vector::populateVectorGatherToConditionalLoadPatterns(patterns);
      // vector.transpose lowering
      // Shuffle16x16 will fallback to Shuffle1D for non 16x16 sizes.
      vector::populateVectorTransposeLoweringPatterns(
          patterns, vector::VectorTransposeLowering::Shuffle16x16);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
    }

    // Unroll load/store from <d1xd2x...xdk> to (d1*d2*...*d(k-1)) slices of
    // <1x1x...x1xdk>.
    {
      RewritePatternSet patterns(&getContext());
      vector::UnrollVectorOptions vectorOptions;
      vectorOptions.setNativeShapeFn(
          [](Operation *op) -> std::optional<SmallVector<int64_t>> {
            auto extractVectorType = [](Operation *op) -> VectorType {
              if (auto loadOp = dyn_cast<vector::LoadOp>(op))
                return loadOp.getVectorType();
              if (auto storeOp = dyn_cast<vector::StoreOp>(op))
                return storeOp.getVectorType();
              return nullptr;
            };

            VectorType vecType = extractVectorType(op);
            if (!vecType)
              return std::nullopt;

            // Only handle rank >= 2 so we actually unroll something.
            int64_t rank = vecType.getRank();
            if (rank < 2)
              return std::nullopt;

            ArrayRef<int64_t> shape = vecType.getShape();
            // Produce native shape: 1 x 1 x ... x (original last dim).
            SmallVector<int64_t> native(rank, 1);
            native.back() = shape.back();
            return native;
          });
      vector::populateVectorUnrollPatterns(patterns, vectorOptions);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        LDBG() << "Unroll failed.";
        return signalPassFailure();
      }
    }

    // Use vector linearization patterns
    {
      MLIRContext &context = getContext();
      TypeConverter converter;
      RewritePatternSet patterns(&context);
      ConversionTarget target(context);
      vector::populateForVectorLinearize(converter, target);
      vector::populateVectorLinearizeBasePatterns(converter, target, patterns);
      vector::populateVectorLinearizeShuffleLikeOpsPatterns(converter, target,
                                                            patterns);
      scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                           target);
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns)))) {
        LDBG() << "Linearization failed.";
        return signalPassFailure();
      }
    }
  }
};
} // namespace
