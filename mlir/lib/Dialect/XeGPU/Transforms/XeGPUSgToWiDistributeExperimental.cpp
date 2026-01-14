//===- XeGPUSgToWiDistributeExperimental.cpp - XeGPU SG to WI Pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUSGTOWIDISTRIBUTEEXPERIMENTAL
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

using namespace mlir;
using namespace mlir::xegpu;

namespace {

struct CreateNdDescOpPattern
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getType();
    // If no layout, nothing to do.
    if (!resultType.getLayout())
      return failure();

    auto newOp = xegpu::CreateNdDescOp::create(
        rewriter, op.getLoc(), resultType.dropLayouts(), op->getOperands(),
        op->getAttrs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct XeGPUSgToWiDistributeExperimentalPass
    : public xegpu::impl::XeGPUSgToWiDistributeExperimentalBase<
          XeGPUSgToWiDistributeExperimentalPass> {
  void runOnOperation() override;
};

} // namespace

void XeGPUSgToWiDistributeExperimentalPass::runOnOperation() {
  // Recover layouts.
  Operation *op = getOperation();
  if (!xegpu::recoverTemporaryLayouts(op)) {
    signalPassFailure();
    return;
  }

  // Define conversion target
  ConversionTarget target(getContext());
  target.addLegalDialect<index::IndexDialect, memref::MemRefDialect,
                         vector::VectorDialect>();
  target.addDynamicallyLegalDialect<xegpu::XeGPUDialect>(
      [](Operation *op) { return true; });

  // Define type converter
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
}

void xegpu::populateXeGPUSgToWiDistributeExperimentalPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<CreateNdDescOpPattern>(typeConverter, patterns.getContext());
}
