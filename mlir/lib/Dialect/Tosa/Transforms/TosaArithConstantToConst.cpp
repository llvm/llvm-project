//===- TosaArithConstantToConst.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that converts tensor-valued arith.constant ops
// into tosa.const so that TOSA pipelines operate on a uniform constant form.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAARITHCONSTANTTOTOSACONSTPASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

/// Returns true when `elementType` is natively representable by tosa.const.
static bool isSupportedElementType(Type elementType) {
  if (isa<FloatType>(elementType))
    return true;

  if (auto intType = dyn_cast<IntegerType>(elementType))
    return intType.isSignless() || intType.isUnsigned();

  if (isa<quant::QuantizedType>(elementType))
    return true;

  return false;
}

class ArithConstantToTosaConst : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constOp,
                                PatternRewriter &rewriter) const override {
    // TOSA constant verification requires a ranked, statically shaped tensor.
    auto resultType = dyn_cast<RankedTensorType>(constOp.getResult().getType());
    if (!resultType || !resultType.hasStaticShape())
      return failure();

    if (!isSupportedElementType(resultType.getElementType()))
      return failure();

    Attribute attr = constOp.getValueAttr();
    auto elementsAttr = dyn_cast<ElementsAttr>(attr);
    if (!elementsAttr)
      return failure();

    auto attrType = dyn_cast<RankedTensorType>(elementsAttr.getType());
    if (!attrType || !attrType.hasStaticShape())
      return failure();

    if (attrType != resultType) {
      // Allow reshape when the payload can be reinterpreted without altering
      // the number of elements or element type. Dense resource attributes
      // cannot be reshaped losslessly, so bail out in that case.
      if (!isa<DenseElementsAttr>(elementsAttr))
        return failure();

      if (attrType.getElementType() != resultType.getElementType())
        return failure();

      auto denseAttr = cast<DenseElementsAttr>(elementsAttr);
      if (denseAttr.getNumElements() != resultType.getNumElements())
        return failure();

      elementsAttr = denseAttr.reshape(resultType);
    }

    auto newConst = tosa::ConstOp::create(rewriter, constOp.getLoc(),
                                          resultType, elementsAttr);
    rewriter.replaceOp(constOp, newConst.getResult());
    return success();
  }
};

struct TosaArithConstantToTosaConstPass
    : public tosa::impl::TosaArithConstantToTosaConstPassBase<
          TosaArithConstantToTosaConstPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ArithConstantToTosaConst>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
