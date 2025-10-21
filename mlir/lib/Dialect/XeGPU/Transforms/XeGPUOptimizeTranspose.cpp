//===- XeGPUOptimizeTranspose.cpp - XeGPU optimize transpose ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUOPTIMIZETRANSPOSE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-optimize-transpose"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {

class XeGPULoadNdPattern final : public OpConversionPattern<xegpu::LoadNdOp> {
public:
  using OpConversionPattern<xegpu::LoadNdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::LoadNdOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return success();
  }
};
} // namespace

void xegpu::populateXeGPUOptimizeTransposePatterns(
    RewritePatternSet &patterns) {
  patterns.add<XeGPULoadNdPattern>(patterns.getContext());
}

namespace {

struct XeGPUOptimizeTransposePass final
    : public xegpu::impl::XeGPUOptimizeTransposeBase<
          XeGPUOptimizeTransposePass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    TypeConverter converter;
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    xegpu::populateXeGPUOptimizeTransposePatterns(patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      DBGS() << "Optimize transpose pass failed.\n";
      return signalPassFailure();
    }
  }
};

} // namespace
